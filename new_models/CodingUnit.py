import torch
import torch.nn as nn

from .BaseModel import BaseModel

class CodingUnit(BaseModel):
    def __init__(self, ch_in_enc, ch_out_enc, ch_LS, prior_network, kernel = 5, stride = 2, ch_in_dec = None,
                 ch_out_dec = None, useCuda=True, maskLatentSpace=False, numOutputChannelsParBranch=0):
        super(CodingUnit, self).__init__()

        if ch_out_dec is None:
            ch_out_dec = ch_in_enc
        if ch_in_dec is None:
            ch_in_dec = ch_out_enc
        padding = int(kernel/2)

        self.maskLatentSpace = maskLatentSpace
        self.resNetToMask = None
        if maskLatentSpace:
            self.resNetToMask = nn.ModuleList()
            self.resNetToMask.append(nn.Conv2d(numOutputChannelsParBranch, 192, kernel_size=1, stride=1))
            if self.training:
                self.resNetToMask.requires_grad_(True)
            self.sigmoid = nn.Sigmoid()

        self.convEnc = nn.Conv2d(ch_in_enc, ch_out_enc, kernel, stride, padding, padding_mode="replicate")

        self.GDNEnc = nn.ReLU(ch_out_enc)

        self.deconvDec = nn.ConvTranspose2d(ch_in_dec + ch_LS, ch_out_dec, kernel, stride, padding, output_padding=stride-1)
        self.GDNDec = nn.ReLU(ch_out_dec)

        self.LSGen = nn.Conv2d(ch_out_enc + ch_in_dec, ch_LS, kernel, stride=1, padding=padding, padding_mode="replicate")

        self.priorNetwork = prior_network
        self.useCuda = useCuda


    def forward_enc(self, x):
        self.encoderSave = self.GDNEnc(self.convEnc(x))
        return self.encoderSave

    def forward_dec(self, y, coder=None, parallelBranch=None):
        LS = self.LSGen(torch.cat((self.encoderSave, y),1))
        mu_LS, sigma_LS, total_bits_prior, total_bits_prior_single = self.priorNetwork(LS, y, coder)
        if parallelBranch is not None:
            alpha = parallelBranch
            for layer in self.resNetToMask:
                alpha = layer(alpha)
            alpha = self.sigmoid(alpha)
            LS = LS - alpha * (LS - mu_LS)

        if self.training:
            noise = torch.nn.init.uniform_(torch.zeros_like(LS), -0.5, 0.5)
            LS_compressed = LS + noise
        else:
            LS_compressed = torch.round(LS)

        if coder is None:
            total_bits_LS, _, information, total_bits_LS_single = self.feature_probs_based_sigma(LS_compressed, mu_LS, sigma_LS)     # estimate required bits
        else:
            data = LS_compressed
            if self.useCuda:
                del LS
                torch.cuda.empty_cache()
                data = LS_compressed.cuda()

            self.encode(data, mu_LS, sigma_LS, coder, mask=None)
            total_bits_LS = 0
            information = torch.zeros_like(y[:,0:1,:,:])
            total_bits_LS_single = 0
        x_hat = self.GDNDec(self.deconvDec(torch.cat((y, LS_compressed),1)))

        return x_hat, total_bits_LS + total_bits_prior, information.sum(dim=1, keepdim=True), total_bits_LS_single + total_bits_prior_single

    def decode_bitstream(self, y, coder, height, width):

        mu_LS, sigma_LS = self.priorNetwork.decode_bitstream(y, coder, height//4, width//4)

        LS_compressed = self.decode(mu_LS, sigma_LS, coder, mask=None)

        x_hat = self.GDNDec(self.deconvDec(torch.cat((y, LS_compressed), 1)))
        return x_hat

    def clearData(self):
        del self.encoderSave







