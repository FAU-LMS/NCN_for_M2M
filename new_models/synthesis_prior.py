#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
from .analysis import Analysis_net
from .analysis_prior import Analysis_prior_net
from .synthesis import Synthesis_net
import math
import torch.nn as nn
import torch

class Synthesis_prior_net(nn.Module):
    '''
    Decode synthesis prior
    '''
    def __init__(self, out_channel_sigma = 128, out_channel_N=16, out_channel_M=16, kernel = 5, strides = 2):
        super(Synthesis_prior_net, self).__init__()
        padding = int(kernel / 2)
        output_padding = strides - 1
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, kernel, stride=1, padding=padding, output_padding=0)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M)))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU()
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, kernel, stride=strides, padding=padding, output_padding=output_padding)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_sigma // 2, kernel, stride=strides, padding=padding, output_padding=output_padding)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_sigma) / (out_channel_sigma + out_channel_sigma))))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, out_channel_sigma // 2, kernel, stride=strides, padding=padding, output_padding=output_padding)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * (out_channel_M + out_channel_sigma) / (out_channel_sigma + out_channel_sigma))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        self.logSigmoid = torch.nn.LogSigmoid()

    def forward(self, x):
        x = self.relu1(self.deconv1(x))
        x = self.relu2(self.deconv2(x))
        return self.deconv3(x), -self.logSigmoid(self.deconv4(x))
        


def build_model():
    input_image = torch.zeros([7,3,256,256])
    analysis_net = Analysis_net()
    analysis_prior_net = Analysis_prior_net()
    synthesis_net = Synthesis_net()
    synthesis_prior_net = Synthesis_prior_net()

    feature = analysis_net(input_image)
    z = analysis_prior_net(feature)

    compressed_z = torch.round(z)

    recon_mu, recon_sigma = synthesis_prior_net(compressed_z)


    compressed_feature_renorm = (feature - recon_mu) / recon_sigma
    compressed_feature_renorm = torch.round(compressed_feature_renorm)
    compressed_feature_denorm = compressed_feature_renorm * recon_sigma + recon_mu

    recon_image = synthesis_net(compressed_feature_denorm)

    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("z : ", z.size())
    print("recon_sigma : ", recon_sigma.size())
    print("recon_image : ", recon_image.size())


if __name__ == '__main__':
    build_model()
