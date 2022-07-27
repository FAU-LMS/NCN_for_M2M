import torch
import torch.nn as nn
import math

import numpy as np
import os
import sys

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.phase = 0
        self.resolution = 1024
        self.encode = self.encode_low_mem
        self.decode = self.decode_low_mem
        self.fnHandleDist = torch.distributions.laplace.Laplace

    def setPhase(self, phase):
        self.phase = phase
        for m in self.modules():
            if m is self:
                continue
            if issubclass(type(m), BaseModel):
                m.setPhase(phase)

    def setApprox(self, approx):
        for m in self.modules():
            if m is self:
                continue
            if issubclass(type(m), BaseModel):
                m.setApprox(approx)

    def feature_probs_based_sigma(self, feature, mu, sigma):
        distribution = self.fnHandleDist(mu, sigma + 1e-10)
        probs = distribution.cdf(feature + 0.5) - distribution.cdf(feature - 0.5)
        information = -1.0 * torch.log(probs + 1e-10) / math.log(2.0)
        total_bits = torch.sum(information)
        return total_bits, probs, information, information.sum((1,2,3))

    def iclr18_estimate_bits_z(self, z):
        probs = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(-1.0 * torch.log(probs + 1e-10) / math.log(2.0))
        return total_bits, probs

    def pdf_to_range(self, pdf, vector, epsilon):
        if epsilon > 0:
            eps = epsilon
        else:
            eps = 0
        res = self.resolution * vector.size(0)
        pdf = pdf.permute((4, 0, 1, 2, 3))
        m1 = (pdf < 1 / res).type(torch.float32)
        m2 = 1 - m1
        s1 = torch.sum(m1 * pdf, dim=0)
        new_val = m1 / res
        s1_new = torch.sum(new_val, dim=0)
        pdf = new_val + m2 * pdf * (1 - s1_new) / (1 - s1 + eps)
        freq = pdf * res
        freq = torch.round(freq)
        freq = freq.type(torch.int)
        freq = freq.permute((1, 2, 3, 4, 0))
        freq = freq.reshape((-1, vector.size(0) - 1))
        cumFreq = torch.zeros((freq.size(0), vector.size(0))).type(torch.int)
        if pdf.is_cuda:
            cumFreq = cumFreq.cuda()
        cumFreq[:, 1:] = torch.cumsum(freq, dim=-1)
        if (epsilon > 0):
            cumFreq[:, -1] = res
        return cumFreq

    def laplace_pdf(self, mu, sigma, tail_mass):
        sigma = sigma.clamp(1e-10, 1e10)
        ln = math.log(tail_mass)
        lower = torch.floor(mu + ln * sigma)
        upper = torch.ceil(mu - ln * sigma)
        l = torch.min(lower)
        u = torch.max(upper)

        laplace = torch.distributions.laplace.Laplace(mu, sigma + 1e-10)
        vector = torch.arange(int(l) - 2, 2 + int(u))
        cdf = torch.zeros(tuple(vector.size()) + tuple(mu.size()))
        if mu.is_cuda:
            cdf = cdf.cuda()
            vector = vector.cuda()
        v = vector.expand(tuple(mu.size()) + tuple(vector.size()))
        v = v.permute((4, 0, 1, 2, 3))
        cdf = laplace.cdf(v + 0.5)
        cdf[0] = 0
        cdf[-1] = 1
        pdf = cdf[1:] - cdf[:-1]
        pdf = pdf.permute((1, 2, 3, 4, 0))

        return pdf, vector

    def laplace_pdf_vector(self, mu, sigma, vector):
        cdf = torch.zeros(tuple(vector.size()) + tuple(mu.size()))
        if mu.is_cuda:
            cdf = cdf.cuda()
        laplace = torch.distributions.laplace.Laplace(mu, sigma + 1e-10)
        v = vector.expand(tuple(mu.size()) + tuple(vector.size()))
        v = v.permute((4, 0, 1, 2, 3))
        cdf = laplace.cdf(v + 0.5)
        cdf[0] = 0
        cdf[-1] = 1
        pdf = cdf[1:] - cdf[:-1]
        pdf = pdf.permute((1, 2, 3, 4, 0))
        return pdf

    def encode_low_mem(self, data, mu_LS, sigma_LS, encoder, mask = None):
        # resolution = vector.size(0) * self.resolution
        if mask is None:
            mask = torch.zeros_like(data, dtype=bool)
        data = data.permute((2, 3, 0, 1))
        shape = np.array(tuple(data.size()))
        for y in range(shape[0]):
            for x in range(shape[1]):
                if mask[0,0,y,x]:
                    continue
                pdf, vector = self.laplace_pdf(mu_LS[:,:,y:y+1,x:x+1], sigma_LS[:,:,y:y+1,x:x+1], 0.01)
                min = vector[0]
                max = vector[-1]
                data_tmp = data[y, x, ...].clone()
                if data_tmp.min() < min:
                    sys.stderr.write("[LSUnit] Warning! Data out of range: Clipping %d to %d (%d cases in total)\n" % (
                        int(data_tmp.min()), min, (data_tmp < min).sum()))
                if data_tmp.max() > max:
                    sys.stderr.write("[LSUnit] Warning! Data out of range: Clipping %d to %d (%d cases in total)\n" % (
                        int(data_tmp.max()), max, (data_tmp > max).sum()))
                data_tmp = data_tmp.clamp(vector[0] + 1, vector[-1])
                data[y, x, ...] = data[y, x, ...].clamp(vector[0] + 1, vector[-1])
                data_tmp = data_tmp - torch.min(vector) - 1
                pdf = pdf.permute((2, 3, 0, 1, 4))
                cumFreq = self.pdf_to_range(pdf, vector, -1)
                data_tmp = torch.reshape(data_tmp, (-1,))
                i = 0
                for d in data_tmp:
                    encoder.encode([int(d)], cumFreq[i].tolist())
                    i += 1
        torch.cuda.empty_cache()

    def encode_fast(self, data, pdf, vector, encoder, mask = None):
        normalized_data = data - torch.min(vector) - 1
        normalized_data = normalized_data.permute((2, 3, 0, 1))
        shape = np.array(tuple(normalized_data.size()))
        normalized_data = torch.reshape(normalized_data, (-1,))
        if mask is None:
            mask = torch.zeros_like(normalized_data, dtype=bool)
        else:
            mask = mask.permute((2,3,0,1))
            mask = torch.reshape(mask, (-1,))
        pdf = pdf.permute((2, 3, 0, 1, 4))

        cumFreq = self.pdf_to_range(pdf, vector, -1)

        del pdf
        self.freeGPU(4)

        normalized_data = normalized_data[torch.logical_not(mask)]
        cumFreq = cumFreq[torch.logical_not(mask)]
        i = 0
        for d in normalized_data:
            encoder.encode([int(d)], cumFreq[i].tolist())
            i += 1
        del cumFreq, normalized_data
        torch.cuda.empty_cache()

    def decode_low_mem(self, mu_LS, sigma_LS, decoder, mask = None):
        shape = [mu_LS.shape[2], mu_LS.shape[3], mu_LS.shape[0], mu_LS.shape[1]]
        device = 'cuda' if self.useCuda else 'cpu'
        normalized_data = torch.zeros(shape, dtype=torch.float32, device=device)

        for y in range(shape[0]):
            for x in range(shape[1]):
                pdf, vector = self.laplace_pdf(mu_LS[:, :, y:y + 1, x:x + 1], sigma_LS[:, :, y:y + 1, x:x + 1], 0.01)
                pdf = pdf.permute((2, 3, 0, 1, 4))
                cumFreq = self.pdf_to_range(pdf, vector, -1)
                i = 0

                for d in cumFreq:
                    data_tmp = decoder.decode(1, d.tolist())
                    normalized_data[y:y+1,x:x+1,0,i] = data_tmp[0]  + torch.min(vector) + 1
                    i += 1

        data = normalized_data
        data = data.permute((2, 3, 0, 1))
        torch.cuda.empty_cache()
        return data


    def decode_fast(self, pdf, vector, decoder, mask):
        pdf = pdf.permute((2, 3, 0, 1, 4))
        shape = [mask.shape[2], mask.shape[3], mask.shape[0], mask.shape[1]]
        cumFreq = self.pdf_to_range(pdf, vector, -1)

        normalized_data = torch.zeros(cumFreq.size(0))

        if mask is None:
            mask = torch.zeros_like(normalized_data, dtype=bool)
        else:
            mask = mask.permute((2,3,0,1))
            mask = torch.reshape(mask, (-1,))

        if pdf.is_cuda:
            normalized_data = normalized_data.cuda()
        for i in range(cumFreq.size(0)):
            if mask[i]:
                continue
            normalized_data[i] = decoder.decode(1, cumFreq[i].tolist())[0]
        data = normalized_data + torch.min(vector) + 1
        data[mask] = 0
        data = torch.reshape(data, tuple(shape))
        data = data.permute((2, 3, 0, 1))
        return data

    def encode_masked(self, values, encoder, cuda, vector):
        z = torch.zeros_like(values)
        minimum = torch.min(vector) + 1
        for h in range(z.shape[2]):
            for w in range(z.shape[3]):
                res_mu_, res_sigma_ = self.autoregressive(z)
                res_mu, res_sigma = res_mu_[:, :, h, w, None, None], res_sigma_[:, :, h, w, None, None]
                pdf = self.laplace_pdf_vector(res_mu, res_sigma, vector)  # ! Change
                pdf = pdf.permute((2, 3, 0, 1, 4))
                cumFreq = self.pdf_to_range(pdf, vector, -1)
                for b in range(z.shape[0]):
                    for c in range(z.shape[1]):
                        z[b, c, h, w] = values[b,c,h,w]
                        encoder.encode([int(values[b,c,h,w]-minimum)], cumFreq[b * z.shape[1] + c].tolist())  # ! Change
        del cumFreq, pdf
        torch.cuda.empty_cache()
        return z

    def decode_masked(self, decoder, cuda, shape, vector):
        z = torch.zeros(tuple(shape))
        if cuda:
            z = z.cuda()
            vector = vector.cuda()
        z = z.permute((2, 3, 0, 1))
        minimum = torch.min(vector) + 1
        for h in range(shape[0]):
            for w in range(shape[1]):
                res_mu, res_sigma = self.autoregressive(z)
                res_mu, res_sigma = res_mu[:, :, h, w, None, None], res_sigma[:, :, h, w, None, None]
                pdf = self.laplace_pdf_vector(res_mu, res_sigma, vector)  # ! Change
                pdf = pdf.permute((2, 3, 0, 1, 4))
                cumFreq = self.pdf_to_range(pdf, vector, -1)
                for b in range(shape[2]):
                    for c in range(shape[3]):
                        z[b, c, h, w] = decoder.decode(1, cumFreq[b * shape[3] + c].tolist())[0] + minimum  # ! Change
        return z

    def read_code_size(self, string):
        size = os.stat(string).st_size * 8
        return torch.tensor(size)

    def save_autoRegressive(self, mu, sigma):
        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()
        np.save(os.path.join(self.root_path, self.file_name + '_masked_mu.npy'), mu)
        np.save(os.path.join(self.root_path, self.file_name + '_masked_sigma.npy'), sigma)

    def load_autoRegressive(self):
        mu = np.load(os.path.join(self.root_path, self.file_name + '_masked_mu.npy'))
        sigma = np.load(os.path.join(self.root_path, self.file_name + '_masked_sigma.npy'))
        return torch.tensor(mu), torch.tensor(sigma)

    def freeGPU(self, id):
        torch.cuda.empty_cache()
