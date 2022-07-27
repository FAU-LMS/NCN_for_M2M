#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5  
# from .basics import *
from .analysis import Analysis_net
import math
import torch.nn as nn
import torch


class Analysis_prior_net(nn.Module):
    '''
    Analysis prior net
    '''
    def __init__(self, input_channels = 64, out_channel_N=16, out_channel_M=16, kernel = 5, strides = 2):
        super(Analysis_prior_net, self).__init__()
        padding = int(kernel / 2)
        self.conv1 = nn.Conv2d(input_channels, out_channel_N, kernel, stride=strides, padding=padding)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (input_channels + out_channel_N) / (2 * input_channels))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(out_channel_N, out_channel_N, kernel, stride=strides, padding=padding)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv2.bias.data, 0.01)
        self.relu2 = nn.LeakyReLU()
        self.conv3 =  nn.Conv2d(out_channel_N, out_channel_M, kernel, stride=1, padding=padding)
        torch.nn.init.xavier_normal_(self.conv3.weight.data, math.sqrt(2))
        torch.nn.init.constant_(self.conv3.bias.data, 0.01)

    def forward(self, x):
        # x = torch.abs(x)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)


def build_model():
    input_image = torch.zeros([5, 3, 256, 256])
    analysis_net = Analysis_net()
    analysis_prior_net = Analysis_prior_net()

    feature = analysis_net(input_image)
    z = analysis_prior_net(feature)
    
    print(input_image.size())
    print(feature.size())
    print(z.size())


if __name__ == '__main__':
    build_model()
