import torch
import torch.nn as nn

class BCConv2D(nn.Module):
    def __init__(self, in_channels_conv, in_channels_bias, out_channels, kernel_size, stride, padding, padding_mode, out_padding, transpose=False):
        if transpose:
            self.conv = nn.ConvTranspose2d(in_channels=in_channels_conv, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, padding_mode=padding_mode, bias=False, output_padding=out_padding)
        else:
            self.conv = nn.Conv2d(in_channels=in_channels_conv, out_channels=out_channels, kernel_size=kernel_size,
                                  stride=stride, padding=padding, padding_mode=padding_mode, bias=False)
        self.bias = nn.Linear(in_features=in_channels_bias, out_features=out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, in_conv, in_bias):
        return self.prelu(self.conv(in_conv)+self.bias(in_bias).view(1,-1,1,1))

    def train_conv(self):
        for p in self.bias.parameters():
            p.requires_grad_(False)
        for p in self.conv.parameters():
            p.requires_grad_(True)

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