#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 00:07:50 2020

@author: alex
"""
import math
import torch.nn as nn
import torch
from .BaseModel import BaseModel
from .logSigmoidApprox import logSigmoidApprox


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0
        self.mask = torch.logical_not(self.mask)

    def forward(self, x):
        # self.weight.data[self.mask] = 0
        self.mask = self.mask.to(x.device)
        self.weight.data = self.weight.data.to(x.device)
        self.bias.data = self.bias.data.to(x.device)
        self.weight.data = self.weight.data * torch.logical_not(self.mask)      # [KF] done to avoid "number of flattened indices did not match number of elements in the value" error
        return super(MaskedConv2d, self).forward(x)


class AutoRegressive(BaseModel):
    """
    Auto regressive net
    """
    def __init__(self, input_channels = 16, out_channel_N=16, out_channel_M=32, kernel = 3):
        super(AutoRegressive, self).__init__()
        padding = int(kernel / 2)
        self.conv1 = MaskedConv2d('A', input_channels, out_channel_N, kernel, stride=1, padding=padding)
        torch.nn.init.xavier_normal_(self.conv1.weight.data, (math.sqrt(2 * (input_channels + out_channel_N) / (2 * input_channels))))
        torch.nn.init.constant_(self.conv1.bias.data, 0.01)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = MaskedConv2d('A', out_channel_N, out_channel_M // 2, kernel, stride=1, padding=padding)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2 * (out_channel_N + out_channel_M) / (out_channel_M + out_channel_M)))
        torch.nn.init.constant_(self.conv2.bias.data, 0.5) #!!
        self.conv3 = MaskedConv2d('A', out_channel_N, out_channel_M // 2, kernel, stride=1, padding=padding)
        torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2 * (out_channel_N + out_channel_M) / (out_channel_M + out_channel_M)))
        torch.nn.init.constant_(self.conv2.bias.data, 1/12) #!!
        self.logSigmoid = nn.LogSigmoid()

    def setApprox(self, approx):
        super(AutoRegressive, self).setApprox(approx)
        #print("Setting approx to " + str(approx) + "(AR)")
        if approx:
            self.logSigmoid = logSigmoidApprox()
        else:
            self.logSigmoid = nn.LogSigmoid()
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        return self.conv2(x), -self.logSigmoid(self.conv3(x))
        