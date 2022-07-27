import torch
import torch.nn as nn
from .GDN import GDN
import sys

from .BaseModel import BaseModel

class Channel(BaseModel):
    def __init__(self, ch_in_enc, ch_out_enc , ch_LS, prior_network, kernel = 5, stride = 2, ch_in_dec = None, ch_out_dec = None):
        super(Channel, self).__init__()

        self.priorNetwork = prior_network

    def forward(self, LS, coder = None):
        if self.training:
            noise = torch.nn.init.uniform_(torch.zeros_like(LS), -0.5, 0.5)
            LS_compressed = LS + noise
        else:
            LS_compressed = torch.round(LS)
        mu_LS, sigma_LS, total_bits_prior = self.priorNetwork(LS, None, coder)
        if coder is None:
            total_bits_LS, _, information = self.feature_probs_based_sigma(LS_compressed, mu_LS, sigma_LS, torch.logical_or(mask_prev, mask_cur))
        else:
            pdf, vector = self.laplace_pdf(mu_LS, sigma_LS, 0.01)
            min = vector[0]
            max = vector[-1]
            if LS_compressed.min() < min:
                sys.stderr.write("[LSUnit] Warning! Data out of range: Clipping %d to %d (%d cases in total)\n" % (int(LS_compressed.min()),min, (LS_compressed<min).sum()))
            if LS_compressed.max() > max:
                sys.stderr.write("[LSUnit] Warning! Data out of range: Clipping %d to %d (%d cases in total)\n" % (int(LS_compressed.max()),max, (LS_compressed>max).sum()))
            LS_compressed = LS_compressed.clamp(vector[0] + 1, vector[-1])
            self.encode(LS_compressed, pdf, vector, coder, torch.zeros_like(LS_compressed, dtype=bool))
            total_bits_LS = 0

        return LS_compressed, total_bits_LS + total_bits_prior

    def decode_bitstream(self, y, mask_prev, mask_cur, coder, height, width):

        mu_LS, sigma_LS = self.priorNetwork.decode_bitstream(y, coder, height//4, width//4)
        mask_prev = mask_prev.repeat((1, mu_LS.shape[1], 1, 1))
        mask_cur = mask_cur.repeat((1, mu_LS.shape[1], 1, 1))

        pdf, vector = self.laplace_pdf(mu_LS, sigma_LS, 0.1)
        LS_compressed = self.decode(pdf, vector, coder, torch.logical_or(mask_prev, mask_cur))

        LS_compressed[mask_prev] = 0 if self.phase < 2 else mu_LS[mask_prev]
        LS_compressed[mask_cur] = 0 if self.phase < 2 else mu_LS[mask_cur]

        x_hat = self.GDNDec(self.deconvDec(torch.cat((y, LS_compressed), 1)))
        return x_hat

    def clearData(self):
        del self.encoderSave







