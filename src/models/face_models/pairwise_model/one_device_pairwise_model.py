import torch.nn as nn
import torch
from torch.nn.parallel import DistributedDataParallel
from ...registry import FACE_MODEL
from ...builder import build_base_model, build_top_model

@FACE_MODEL.register_module
class OneDevicePairwiseModel(nn.Module):
    def __init__(self, pretrained, base_model, top_model, base_model_gpus, top_model_gpu):
        super(OneDevicePairwiseModel, self).__init__()
        self.base_model = build_base_model(base_model).to('cuda:{}'.format(base_model_gpus[0]))
        self.top_model = build_top_model(top_model).to('cuda:{}'.format(top_model_gpu))
        self.init_weights(pretrained)
        # self.base_model = nn.DataParallel(self.base_model, device_ids=base_model_gpus, output_device=base_model_gpus[0])
        self.base_model = DistributedDataParallel(self.base_model, device_ids=base_model_gpus,
                                                  output_device=base_model_gpus[0])

    def init_weights(self, pretrained=None):
        self.base_model.init_weights(pretrained)
        self.top_model.init_weights()

    def _forward_train(self, data):
        input = data['img']
        label = data['label']
        output = self.base_model(input)
        loss, sp, sn = self.top_model(output, label, mode='train')
        loss = loss.mean()
        loss.backward()
        return {'loss': loss, 'sp': sp, 'sn': sn}

    def _forward_val(self, data):
        pass

    def forward(self, data, mode='train'):
        if mode == 'train':
            re = self._forward_train(data)
            return re
        elif mode == 'val':
            NotImplementedError

    def train(self, mode=True):
        self.base_model.train(mode)
        self.top_model.train(mode)
