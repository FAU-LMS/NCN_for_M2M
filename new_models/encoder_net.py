import math
import torch.nn as nn
import torch
from .GDN import GDN
from .BCConv import BCConv2D


class Encoder_Net(nn.Module):
    '''
    Analysis net
    '''
    def __init__(self, num_layers = 4, channels = [3,192,192,192,128], kernel = 5, strides = 2, masked = False):
        super(Encoder_Net, self).__init__()
        assert num_layers == len(channels) - 1
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        if type(strides) != list:
            strides = [strides]*num_layers
        padding = int(kernel/2)
        for i in range(num_layers):
            self.layers.append(nn.Conv2d(channels[i], channels[i+1], kernel, stride=strides[i], padding=padding))
            torch.nn.init.xavier_normal_(self.layers[i].weight.data,
                                         (math.sqrt(2 * (channels[i] + channels[i+1]) / (2 * channels[i]))))
            torch.nn.init.constant_(self.layers[i].bias.data, 0.01)
            if i != num_layers - 1:
                self.activations.append(GDN(channels[i+1]))
        if masked:
            self.forward = self.forward_masked
        else:
            self.forward = self.forward_unmasked



    def forward_unmasked(self, x):
        for i in range(len(self.layers)-1):
            x = self.activations[i](self.layers[i](x))
        return self.layers[-1](x)

    def forward_masked(self, x, masks):
        for i in range(len(self.layers)-1):
            x = self.activations[i](self.layers[i](x) * masks[i])
        return self.layers[-1](x) * masks[-1]

class EncoderNetBC(nn.Module):
    def __init__(self, num_layers = 4, channels = [3,192,192,192,128], kernel = 5, strides = 2, masked = False, biasInput = 64):
        super(EncoderNetBC, self).__init__()
        assert num_layers == len(channels) - 1
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        if type(strides) != list:
            strides = [strides] * num_layers
        padding = int(kernel / 2)
        for i in range(num_layers):
            self.layers.append(BCConv2D(channels[i], biasInput, channels[i + 1], kernel, stride=strides[i], padding=padding, padding_mode="replicate"))
            torch.nn.init.xavier_normal_(self.layers[i].weight.data,
                                         (math.sqrt(2 * (channels[i] + channels[i + 1]) / (2 * channels[i]))))
            torch.nn.init.constant_(self.layers[i].bias.data, 0.01)
            if i != num_layers - 1:
                self.activations.append(GDN(channels[i + 1]))

    def forward(self, x, b):
        for i in range(len(self.layers) - 1):
            x = self.activations[i](self.layers[i](x,b))
        return self.layers[-1](x)

    def train_conv(self):
        for l in self.layers:
            l.train_conv()

    def train_bias(self):
        for p in self.bias.parameters():
            p.requires_grad_(True)
        for p in self.conv.parameters():
            p.requires_grad_(False)

    def train_both(self):
        for p in self.bias.parameters():
            p.requires_grad_(True)
        for p in self.conv.parameters():
            p.requires_grad_(True)






def build_model():
    input_image = torch.zeros([4, 3, 256, 256])

    analysis_net = Analysis_net()
    feature = analysis_net(input_image)

    print(feature.size())


if __name__ == '__main__':
    build_model()