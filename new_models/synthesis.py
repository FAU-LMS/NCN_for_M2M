#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch
from .GDN import GDN
from .analysis import Analysis_net

class Synthesis_net(nn.Module):
    '''
    Decode synthesis
    '''
    def __init__(self, out_channel_last = 3, out_channel_N=16, out_channel_M=64, kernel = 5, strides = 2, activation = "GDN"):
        super(Synthesis_net, self).__init__()
        padding = int(kernel / 2)
        output_padding = strides - 1
        self.deconv1 = nn.ConvTranspose2d(out_channel_M, out_channel_N, kernel, stride=strides, padding=padding, output_padding=output_padding)
        torch.nn.init.xavier_normal_(self.deconv1.weight.data, (math.sqrt(2 * 1 * (out_channel_M + out_channel_N) / (out_channel_M + out_channel_M))))
        torch.nn.init.constant_(self.deconv1.bias.data, 0.01)
        self.igdn1 = GDN(out_channel_N, inverse=True)
        self.deconv2 = nn.ConvTranspose2d(out_channel_N, out_channel_N, kernel, stride=strides, padding=padding, output_padding=output_padding)
        torch.nn.init.xavier_normal_(self.deconv2.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv2.bias.data, 0.01)
        self.igdn2 = GDN(out_channel_N, inverse=True)
        self.deconv3 = nn.ConvTranspose2d(out_channel_N, out_channel_N, kernel, stride=strides, padding=padding, output_padding=output_padding)
        torch.nn.init.xavier_normal_(self.deconv3.weight.data, math.sqrt(2 * 1))
        torch.nn.init.constant_(self.deconv3.bias.data, 0.01)
        self.igdn3 = GDN(out_channel_N, inverse=True)
        self.deconv4 = nn.ConvTranspose2d(out_channel_N, out_channel_last, kernel, stride=strides, padding=padding, output_padding=output_padding)
        torch.nn.init.xavier_normal_(self.deconv4.weight.data, (math.sqrt(2 * (out_channel_N + out_channel_last) / (out_channel_N + out_channel_N))))
        torch.nn.init.constant_(self.deconv4.bias.data, 0.01)
        if (activation == "LeakyReLU"):
            self.igdn1 = nn.LeakyReLU()
            self.igdn2 = nn.LeakyReLU()
            self.igdn3 = nn.LeakyReLU()
        
    def forward(self, x):
        x = self.igdn1(self.deconv1(x))
        x = self.igdn2(self.deconv2(x))
        x = self.igdn3(self.deconv3(x))
        x = self.deconv4(x)
        return x

# synthesis_one_pass = tf.make_template('synthesis_one_pass', synthesis_net)

def build_model():
    input_image = torch.zeros([7,3,256,256])
    analysis_net = Analysis_net()
    synthesis_net = Synthesis_net()
    feature = analysis_net(input_image)
    recon_image = synthesis_net(feature)

    print("input_image : ", input_image.size())
    print("feature : ", feature.size())
    print("recon_image : ", recon_image.size())

# def main(_):
#   build_model()


if __name__ == '__main__':
    build_model()
