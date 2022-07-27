import torch
import torch.nn as nn
import math
from new_models import *
from pytorch_msssim import MS_SSIM
from range_coder import RangeEncoder, RangeDecoder      # https://github.com/lucastheis/rangecoder


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class CodecNcn(BaseModel):
    def __init__(self, ch_top, ch_pass, ch_LS, path=None, useCuda=True,
                 requiresSSIM=True,
                 parallelBranch=None, rcnnMean=0, rcnnStd=0):
        super(CodecNcn, self).__init__()

        assert len(ch_pass) == len(ch_LS)

        self.useCuda = useCuda

        self.parallelBranch = parallelBranch
        numOutputChannelsParBranch = None
        maskLatentSpace = False
        if parallelBranch:
            maskLatentSpace = True
            # estimate the number of output channels from model
            if type(parallelBranch) == torch.nn.modules.container.ModuleList:       # for ResNet50 FPN from Mask R-CNN it is known
                numOutputChannelsParBranch = 1024
            else:
                if hasattr(parallelBranch[-1], 'out_channels'):
                    numOutputChannelsParBranch = parallelBranch[-1].out_channels
                else:
                    if hasattr(parallelBranch[-2], 'out_channels'):
                        numOutputChannelsParBranch = parallelBranch[-2].out_channels
            if numOutputChannelsParBranch == None:
                raise RuntimeError('Cannot estimate number of output channels of parallel branch!')

            self.pixelMean = rcnnMean
            self.pixelStd = rcnnStd

        kernel = 5
        padding = int(kernel/2)
        padding_out = 1

        self.encoder = nn.ModuleList()          # conv layers
        self.encoder_GDN = nn.ModuleList()      # non-linearities
        self.decoder = nn.ModuleList()
        self.decoder_GDN = nn.ModuleList()

        ch_dec_top = ch_top[::-1] + [3]
        # reduce number of channels in latent space such that DV channels can be added from parallel branch
        if parallelBranch:
            ch_enc_in = [3, 192, 192]
            ch_enc_out = [192, 192, 192]
        else:
            ch_enc_in = [3] + ch_top
            ch_enc_out = ch_top

        for i in range(len(ch_top)):
            self.encoder.append(nn.Conv2d(ch_enc_in[i], ch_enc_out[i], kernel, stride=2, padding=padding, padding_mode='replicate'))
            self.encoder_GDN.append(nn.ReLU())

            self.decoder.append(
                nn.ConvTranspose2d(ch_dec_top[i], ch_dec_top[i + 1], kernel, stride=2, padding=padding, output_padding=padding_out))

            if i < len(ch_top)-1:
                self.decoder_GDN.append(nn.ReLU())

        self.LSUnits = nn.ModuleList()
        ch_pass = [ch_top[-1]] + ch_pass
        for i in range(len(ch_pass)-1):
            p = PriorCoder([ch_LS[i],ch_LS[i]//2] , ch_pass[i+1],ch_LS[i]//2, kernel, useCuda=useCuda)
            self.LSUnits.append(CodingUnit(ch_pass[i], ch_pass[i + 1], ch_LS[i], p, useCuda=useCuda,
                                           maskLatentSpace=maskLatentSpace, numOutputChannelsParBranch=numOutputChannelsParBranch))

        self.requiresSSIM = requiresSSIM
        if self.requiresSSIM:
            self.MSSSIM = MS_SSIM(1.0)
            self.MSSSIM_single = MS_SSIM(1.0, size_average=False)

        self.path = path
        self.results = {}

    def forward(self, imgData):
        img = imgData
        height_orig = img.shape[2]
        width_orig = img.shape[3]
        img_orig = img

        if self.path is None:
            coder = None
        else:
            coder = RangeEncoder(self.path)

            bits_height = [int(i) for i in list("{0:016b}".format(height_orig))]
            assert len(bits_height) == 16
            bits_width = [int(i) for i in list("{0:016b}".format(width_orig))]
            assert len(bits_width) == 16
            fFile = open('encoder_log.txt', 'w')
            fFile.close()

            coder.encode(bits_height + bits_width, [0,1,2])


        f = img
        f_parMulRes4 = None
        f_parRes4 = None
        if self.parallelBranch:
            if self.pixelMean[0][0][0] > 1:
                f_preprocessed = (f*255 - self.pixelMean) / self.pixelStd       #  and shifted by pre-trained parameters of Mask R-CNN for parallel branch
            else:
                f_preprocessed = (f - self.pixelMean) / self.pixelStd       #  and shifted by pre-trained parameters of Mask R-CNN for parallel branch

            if len(self.parallelBranch) == 4:
                ## forward ResNet
                assert len(self.parallelBranch) == (len(self.encoder) + len(self.LSUnits))
                f_par = self.parallelBranch[0](f_preprocessed)
                f_parRes2 = self.parallelBranch[1](f_par)
                f_parRes3 = self.parallelBranch[2](f_parRes2)
                f_parRes4 = self.parallelBranch[3](f_parRes3)
            elif len(self.parallelBranch) == 30:
                ## forward VGG 16
                assert self.maskLatentSpace, 'VGG 16 is currently only supported with masking the latent space'
                f_parRes4 = self.parallelBranch(f_preprocessed)
            else:
                raise RuntimeError('Other parallel branches are currently not implemented')

        for i in range(len(self.encoder)):
            f = self.encoder_GDN[i](self.encoder[i](f))

        if not self.training:
            f = f.detach()
            self.freeGPU(2)

        for i in range(len(self.LSUnits)):
            f = self.LSUnits[i].forward_enc(f)

        bit_map_acc = torch.zeros((f.shape[0],1,f.shape[2], f.shape[3]), device=f.device)

        total_rate = torch.zeros((1), device=img.device)
        total_rate_single = torch.zeros((img.shape[0]), device=img.device)

        d = torch.zeros_like(f)
        if not self.training:
            del f
            self.freeGPU(0)

        for i in range(len(self.LSUnits)):
            d, total_bits, bit_map, total_bits_single = self.LSUnits[-1 - i].forward_dec(d, coder, f_parRes4)

            bit_map_acc += bit_map

            rate = total_bits/(img.shape[0]*img.shape[2]*img.shape[3])
            total_rate += rate
            total_rate_single += total_bits_single/(img.shape[0]*img.shape[2]*img.shape[3])

        if coder is not None:
            coder.close()
            total_bits = self.read_code_size(self.path)
            total_rate = total_bits/(img.shape[0]*img.shape[2]*img.shape[3])

        if not self.training:
            for ls in self.LSUnits:
                ls.clearData()
            self.freeGPU(1)

        for i in range(len(self.decoder)-1):
            d = self.decoder_GDN[i](self.decoder[i](d))

        rec = self.decoder[-1](d)

        rec = rec[:,:,:height_orig, :width_orig]

        if self.requiresSSIM:
            ms_ssim_loss = 1-self.MSSSIM(img_orig, rec)
            ms_ssim_loss_single = (1-self.MSSSIM_single(img_orig, rec))/img.shape[0]
        else:
            ms_ssim_loss = torch.zeros(1)
            ms_ssim_loss_single = torch.zeros(1)
        mse = ((img_orig-rec)**2).mean()
        mse_single = ((img_orig-rec)**2).mean((1,2,3))/img.shape[0]

        return ms_ssim_loss, mse, total_rate, rec

    def decode_bitstream(self, path):
        f = open('decoder_log.txt', 'w')
        f.close()

        coder = RangeDecoder(path)

        height_orig = int("".join([str(i) for i in coder.decode(16, [0,1,2])]),2)
        width_orig = int("".join([str(i) for i in coder.decode(16, [0,1,2])]),2)

        height = math.ceil(height_orig / 256) * 256
        width = math.ceil(width_orig / 256) * 256

        self.device = 'cuda' if self.useCuda else 'cpu'
        d = torch.zeros((1,192,height//16, width//16), device=self.device)

        for i in range(len(self.LSUnits)):
            d = self.LSUnits[-1 - i].decode_bitstream(d, coder, height // 16 * 2 ** i, width // 16 * 2 ** i)

        for i in range(len(self.decoder)-1):
            d = self.decoder_GDN[i](self.decoder[i](d))

        rec = self.decoder[-1](d)

        rec = rec[:,:,:height_orig, :width_orig]

        return rec

    def calc_feature_dist(self, orig, rec, sad=True, keyFeatureSpace='p2', normalize=True, croppingRange=16):
        featureOrig = self.evaluationNet.backbone(orig * 255)     # w/o 255, nothing would be detected
        featureOrig = featureOrig[keyFeatureSpace]
        featureRec = self.evaluationNet.backbone(rec * 255)
        featureRec = featureRec[keyFeatureSpace]

        # from misc.helper import plotFeature
        # plotFeature(featureOrig, 0)
        # plt.figure()
        # plotImage(orig, 0)
        # # plt.show()

        dif = featureOrig - featureRec
        dif = dif[:,:,croppingRange:-croppingRange,croppingRange:-croppingRange]
        if sad:
            retTen = torch.abs(dif).sum()       # SAD
        else:
            retTen = torch.square(dif).sum()    # SSE
        retTen /= 255
        if normalize:
            retTen = retTen / orig.shape[1] / orig.shape[2] / orig.shape[3]     # normalize to feature pixel
        return retTen

    def freeGPU(self, id):
        torch.cuda.empty_cache()