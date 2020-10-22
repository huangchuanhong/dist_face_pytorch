import math

import torch.nn as nn
import torch.utils.checkpoint as cp

from ...utils import constant_init, kaiming_init
from ....runner import load_checkpoint
from ...utils import build_norm_layer
from .regnet import Bottleneck as Bottleneck_
from .regnet import make_res_layer, RegNet

class Bottleneck(Bottleneck_):
    expansion = 1

    def __init__(self,
                 *args, **kwargs):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__(*args, **kwargs)       
        self.relu = nn.PReLU(self.planes)
        # self.relu = nn.ReLU(inplace=True)

class RegNetP(RegNet):
    """RegNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        num_stages (int): Resnet stages, normally 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        frozen_stages (int): Stages to be frozen (all param fixed). -1 means
            not freezing any parameters.
        normalize (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): whether to use zero init for last norm layer
            in resblocks to let them behave as identity.
    """
    """
    GFlops groups not equals the groups of paper because of nnie
    """
    arch_settings = {
        '3.2GF': {'depth':[2, 6, 15, 2],
                  'width':[96, 192, 432, 1008]},
 
    }

    def __init__(self,
                 **kwargs):
        super(RegNetP, self).__init__(**kwargs)
        self.block = Bottleneck
        self.inplanes = 64

        self._make_stem_layer()

        self.res_layers = []
        for i, num_blocks in enumerate(self.stage_blocks):
            stride = self.strides[i]
            dilation = self.dilations[i]
            res_layer = make_res_layer(
                self.block,
                self.inplanes,
                self.stage_planes[i],
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=self.with_cp,
                normalize=self.normalize)
            self.inplanes = self.stage_planes[i]
            layer_name = 'layer{}'.format(i + 1)
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)

    def _make_stem_layer(self):
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1_name, norm1 = build_norm_layer(
            self.normalize, 64, postfix=1)
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.PReLU(64)
        # self.relu = nn.ReLU()