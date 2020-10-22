import math

import torch.nn as nn
import torch
import torch.utils.checkpoint as cp

from ...utils import constant_init, kaiming_init
from ....runner import load_checkpoint
from ...utils import build_norm_layer
from .regnet import make_res_layer, RegNet
import torch.nn.functional as F
from .layers import ECALayer

class ECABottleneck(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 group_width=48,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 normalize=dict(type='BN')):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(ECABottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        self.inplanes = inplanes
        self.planes = planes
        self.normalize = normalize
        self.group_width = group_width
        if style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(normalize, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(normalize, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            normalize, planes * self.expansion, postfix=3)

        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=self.conv2_stride,
            groups=planes//self.group_width,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm2_name, norm2)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.add_module(self.norm3_name, norm3)

        self.eca = ECALayer(planes * self.expansion)

        # downsample
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp
        self.normalize = normalize

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.norm3(out)

            out = self.eca(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            # out = torch.mul(out, out1)
            out = torch.add(out, identity)
            #out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out

class ECARegNet(RegNet):
    arch_settings = {
        '3.2GF': {'depth':[2, 6, 15, 2],
                  'width':[96, 192, 432, 1008]},
        '6.4GF': {'depth':[2, 4, 10, 1],
                  #'width':[168, 392, 784, 1624], origin},
                  'width':[144, 384, 768, 1632]},
        '12GF': {'depth':[2, 5, 11, 1],
                 'width':[240, 432, 912, 2256]},
    }

    def __init__(self,
                 gflops,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 frozen_stages=-1,
                 normalize=dict(type='BN', frozen=False),
                 norm_eval=False,
                 with_cp=False,
                 zero_init_residual=True):
        super(RegNet, self).__init__()
        if gflops not in self.arch_settings:
            raise KeyError('invalid gflops {} for regnet'.format(gflops))
        self.gflops = gflops
        self.block = ECABottleneck
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.frozen_stages = frozen_stages
        self.normalize = normalize
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.zero_init_residual = zero_init_residual

        self.stage_blocks = self.arch_settings[gflops]['depth'][:num_stages]
        self.stage_planes = self.arch_settings[gflops]['width'][:num_stages]
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                self.stage_planes[i],
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
               normalize=normalize)
            self.inplanes = self.stage_planes[i]
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

