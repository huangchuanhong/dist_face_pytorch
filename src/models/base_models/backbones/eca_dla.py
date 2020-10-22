#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from https://github.com/ucbdrive/dla/blob/master/dla.py

import math
from os.path import join

import torch
from torch import nn
import torch.utils.model_zoo as model_zoo
from .layers import ECALayer

BatchNorm = nn.BatchNorm2d

WEB_ROOT = 'http://dl.yf.io/dla/models'


# def get_model_url(data, name):
#     return join(WEB_ROOT, data.name,
#                 '{}-{}.pth'.format(name, data.model_hash[name]))


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(Bottleneck, self).__init__()
        expansion = Bottleneck.expansion
        bottle_planes = planes // expansion
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)

        self.eca = ECALayer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        out = out + residual
        out = self.relu(out)

        return out

class BottleneckX(nn.Module):
    expansion = 2
    group_channels = 16

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        assert(planes % BottleneckX.group_channels == 0)
        bottle_planes = planes
        cardinality = bottle_planes // BottleneckX.group_channels
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        #bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.eca = ECALayer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.eca(out)

        out = out + residual
        out = self.relu(out)

        return out

class BottleneckX_origin(nn.Module):
    expansion = 2
    cardinality = 32

    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BottleneckX, self).__init__()
        cardinality = BottleneckX.cardinality
        # dim = int(math.floor(planes * (BottleneckV5.expansion / 64.0)))
        # bottle_planes = dim * cardinality
        bottle_planes = planes * cardinality // 32
        self.conv1 = nn.Conv2d(inplanes, bottle_planes,
                               kernel_size=1, bias=False)
        self.bn1 = BatchNorm(bottle_planes)
        self.conv2 = nn.Conv2d(bottle_planes, bottle_planes, kernel_size=3,
                               stride=stride, padding=dilation, bias=False,
                               dilation=dilation, groups=cardinality)
        self.bn2 = BatchNorm(bottle_planes)
        self.conv3 = nn.Conv2d(bottle_planes, planes,
                               kernel_size=1, bias=False)
        self.bn3 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = out + residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x

class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False, project=True):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1 and (self.levels == 1 or self.level_root):
            #self.downsample = nn.MaxPool2d(stride, stride=stride)
            self.downsample = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=stride, bias=False)
            # self.downsample = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        if in_channels != out_channels and project and self.levels == 1:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children
        bottom = self.downsample(x) if self.downsample else x
        if self.levels == 1:
            residual = self.project(bottom) if self.project else bottom
        else:
            residual = None
        if self.level_root:
            children.append(bottom)
        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class ECADLA(nn.Module):
    def __init__(self, levels, channels,
                 block=BasicBlock, residual_root=False, return_levels=False):
        super(ECADLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        if isinstance(block, str):
            if block == 'BasicBlock':
                block = BasicBlock
            elif block == 'Bottleneck':
                block = Bottleneck
            elif block == 'BottleneckX':
                block = BottleneckX
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=2,
                      padding=1, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))
        # self.level0 = self._make_conv_level(
        #     channels[0], channels[0], levels[0])
        # self.level1 = self._make_conv_level(
        #     channels[0], channels[1], levels[1], stride=2)
        self.level2 = Tree(levels[0], block, channels[0], channels[1], 1,
                           level_root=False,
                           root_residual=residual_root)
        self.level3 = Tree(levels[1], block, channels[1], channels[2], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[2], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[3], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)

        # self.pool_ = nn.AdaptiveAvgPool2d((7, 7))
        # # self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
        # #                     stride=1, padding=0, bias=True)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(2, 6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            return y[-1]
        #     # x = self.avgpool(x)
        #     # x = self.fc(x)
        #     # x = x.view(x.size(0), -1)
        #
        #     # return x

    # def load_pretrained_model(self, data_name, name):
    #     assert data_name in dataset.__dict__, \
    #         'No pretrained model for {}'.format(data_name)
    #     data = dataset.__dict__[data_name]
    #     fc = self.fc
    #     if self.num_classes != data.classes:
    #         self.fc = nn.Conv2d(
    #             self.channels[-1], data.classes,
    #             kernel_size=1, stride=1, padding=0, bias=True)
    #     try:
    #         model_url = get_model_url(data, name)
    #     except KeyError:
    #         raise ValueError(
    #             '{} trained on {} does not exist.'.format(data.name, name))
    #     self.load_state_dict(model_zoo.load_url(model_url))
    #     self.fc = fc

    def init_weights(self, pretrained=None):
        assert(pretrained==None)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
