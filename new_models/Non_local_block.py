"""
Non-local block (NLB) as implemented in https://github.com/NJUVISION/NIC/blob/main/code/Model/basic_module.py from the Chen 2021 paper
NLB is from Wang et al. 2017 Non-local Neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as f


class Non_local_Block(nn.Module):
    def __init__(self, in_channel, out_channel, s=1):
        super(Non_local_Block, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.g = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.theta = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.phi = nn.Conv2d(self.in_channel, self.out_channel, 1, 1, 0)
        self.W = nn.Conv2d(self.out_channel, self.in_channel, 1, 1, 0)
        self.s = s
        self.downscaling = nn.MaxPool2d(s)
        nn.init.constant(self.W.weight, 0)
        nn.init.constant(self.W.bias, 0)

    def forward(self, x):
        # x_size: (b c h w)

        batch_size = x.size(0)
        g_x = self.g(x)
        if self.s > 1:  # downscale features as described by Chen et al.
            g_x = self.downscaling(g_x)
        g_x = g_x.view(batch_size, self.out_channel, -1).permute(0, 2, 1)
        theta_x = self.theta(x).view(batch_size, self.out_channel, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x)
        if self.s > 1:  # downscale features as described by Chen et al.
            phi_x = self.downscaling(phi_x)
        phi_x = phi_x.view(batch_size, self.out_channel, -1)

        f1 = torch.matmul(theta_x, phi_x)
        f_div_C = f.softmax(f1, dim=-1)
        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.out_channel, *x.size()[2:])
        W_y = self.W(y)
        z = W_y+x

        return z