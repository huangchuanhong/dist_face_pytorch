import torch
import torch.nn as nn
import math

class SELayer(nn.Module):
    def __init__(self, planes, reduction=16):
        super(SELayer, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(
            planes, planes // reduction, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // reduction, planes, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, input):
        out = self.global_pool(input)
        out = self.conv_down(out)
        out = self.relu(out)
        out = self.conv_up(out)
        out = self.sig(out)
        return out

class ECALayer(nn.Module):
    def __init__(self, planes, gamma=2, b=1):
        super(ECALayer, self).__init__()
        t = int(abs(math.log(planes, 2) + 1) / gamma)
        k = t if t % 2 else t + 1

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(1, 1, kernel_size=(k, 1), padding=(int(k / 2), 0), bias=False)
        self.sig = nn.Sigmoid()

        self.planes = planes

    def forward(self, input):
        out = self.global_pool(input)
        out = out.view(-1, 1, self.planes, 1)
        out = self.conv(out)
        out = out.view(-1, self.planes, 1, 1)
        out = self.sig(out)
        out = torch.mul(input, out)
        return out