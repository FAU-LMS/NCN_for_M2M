import torch
import torch.nn as nn
from .GDN import GDN
from .autoRegressive import AutoRegressive
from .BaseModel import BaseModel
from.logSigmoidApprox import logSigmoidApprox
import sys
import os

class PriorCoder(BaseModel):
    def __init__(self, layers_coder, layers_condition, layers_LS, kernel, useCuda=True):
        super(PriorCoder, self).__init__()

        padding = int(kernel/2)

        self.device = 'cuda' if useCuda else 'cpu'

        self.encoder_first = nn.Conv2d(layers_coder[0]+layers_condition, layers_coder[0], kernel, stride=2, padding=padding, padding_mode="replicate", device=self.device)

        self.encoder = nn.ModuleList()
        self.encoder_GDN = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.decoder_GDN = nn.ModuleList()

        for i in range(len(layers_coder)-1):
            self.encoder_GDN.append(GDN(layers_coder[i]))
            self.encoder.append(nn.Conv2d(layers_coder[i], layers_coder[i+1], kernel, stride=2, padding=padding, padding_mode="replicate", device=self.device))

        layers_decoder = ([layers_coder[0]*2] + layers_coder)[::-1]

        for i in range(len(layers_decoder) - 1):
            self.decoder.append(nn.ConvTranspose2d(layers_decoder[i], layers_decoder[i+1], kernel, stride=2, padding=padding, output_padding=1, device=self.device))
            self.decoder_GDN.append(GDN(layers_decoder[i + 1], inverse=True))

        self.decoder_final = nn.Conv2d(layers_coder[0]*2+layers_condition, layers_coder[0]*2, kernel, stride=1, padding=padding, padding_mode='replicate', device=self.device)

        self.autoregressive = AutoRegressive(layers_coder[-1], layers_coder[-1], layers_coder[-1]*2)
        self.logSigmoid = nn.LogSigmoid()
        self.relu = torch.nn.ReLU()
        self.useCuda = useCuda

    def setApprox(self, approx):
        super(PriorCoder, self).setApprox(approx)
        #print("Setting approx to " + str(approx)  + "(PC)")
        if approx:
            self.logSigmoid = logSigmoidApprox()
        else:
            self.logSigmoid = nn.LogSigmoid()

    def forward(self, LS, cond, coder):
        if cond is None:
            f = self.encoder_first(LS)
        else:
            f = self.encoder_first(torch.cat((LS, cond),1))
        for i in range(len(self.encoder)):
            f = self.encoder[i](self.encoder_GDN[i](f)) # 'Falsche' Reihenfolge von Activation und Conv ist richtig!
        if self.training:
            noise = torch.nn.init.uniform_(torch.zeros_like(f), -0.5, 0.5)
            f_compressed = f + noise
        else:
            f_compressed = torch.round(f)
        if coder is None:
            mu_f, sigma_f = self.autoregressive(f_compressed)
            total_bits_f, _, _, total_bits_f_single = self.feature_probs_based_sigma(f_compressed, mu_f, sigma_f)
        else:
            min = -55
            max = 55
            vector = torch.arange(min, max+1)
            if self.useCuda:
                vector = vector.cuda()
            if f_compressed.min() < min:
                sys.stderr.write("Warning! Data out of range: Clipping %d to %d (%d cases in total)\n" % (int(f_compressed.min()),min, (f_compressed<min).sum()))
            if f_compressed.max() > max:
                sys.stderr.write("Warning! Data out of range: Clipping %d to %d (%d cases in total)\n" % (int(f_compressed.max()),max, (f_compressed>max).sum()))
            f_compressed = f_compressed.clamp(vector[0] + 1, vector[-1])
            self.encode_masked(f_compressed, coder, True, vector)
            total_bits_f = 0
            total_bits_f_single = 0

        d = f_compressed
        for i in range(len(self.decoder)):
            d = self.decoder_GDN[i](self.decoder[i](d))
        r = self.decoder_final(torch.cat((d, cond),1))
        mu = r[:,:r.shape[1]//2,:,:]
        sigma = -self.logSigmoid(r[:, r.shape[1] // 2:, :, :])
        return mu, sigma, total_bits_f, total_bits_f_single

    def decode_bitstream(self, cond, coder, height, width):
        vector = torch.arange(-55, 56)
        if self.useCuda:
            vector = vector.cuda()
        d = self.decode_masked(coder, self.useCuda, ( height, width, 1, self.decoder[0].weight.shape[0]), vector)

        for i in range(len(self.decoder)):
            d = self.decoder_GDN[i](self.decoder[i](d))
        r = self.decoder_final(torch.cat((d, cond), 1))
        mu = r[:, :r.shape[1] // 2, :, :]
        sigma = -self.logSigmoid(r[:, r.shape[1] // 2:, :, :])
        return mu, sigma

