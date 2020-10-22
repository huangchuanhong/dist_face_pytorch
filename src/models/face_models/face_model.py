import torch.nn as nn
import torch
# import apex.parallel.DistributedDataParallel as DistributedDataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from ..registry import FACE_MODEL
from ..builder import build_base_model, build_top_model

import logging
import time
_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=_format, level='DEBUG')
_logger = logging.getLogger('debug')



@FACE_MODEL.register_module
class FaceModel(nn.Module):
    def __init__(self, pretrained, base_model, top_model, base_model_gpus, top_model_gpus):
        super(FaceModel, self).__init__()
        self.base_model = build_base_model(base_model).to('cuda:{}'.format(base_model_gpus[0]))
        self.top_model = build_top_model(top_model).to('cuda:{}'.format(top_model_gpus[0]))
        self.init_weights(pretrained)
        print('base_model={}'.format(self.base_model))
        print('top_model={}'.format(self.top_model))
        #self.base_model = nn.DataParallel(self.base_model, device_ids=base_model_gpus, output_device=base_model_gpus[0])
        self.base_model = DistributedDataParallel(self.base_model, device_ids=base_model_gpus, output_device=base_model_gpus[0])
        self.top_model = DistributedDataParallel(self.top_model, device_ids=top_model_gpus, output_device=top_model_gpus[0])
        #self.top_model = nn.DataParallel(self.top_model, device_ids=top_model_gpus, output_device=top_model_gpus[0])
     
    def init_weights(self, pretrained=None):
        self.base_model.init_weights(pretrained)
        self.top_model.init_weights()

    def _forward_train(self, data):
        input = data['img']
        label = data['label']
        output = self.base_model(input)
        loss = self.top_model(output, label, mode='train')
        loss = loss.mean()
        loss.backward()
        for n, p in self.named_parameters():
            if p.grad is None:
                print(n)
        #def _average_gradients(model):
        #    size = float(dist.get_world_size())
        #    for name, param in model.named_parameters():
        #        print('param.grad.data={}'.format(param.grad.data))
        #        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        #        param.grad.data /= size
        #_average_gradients(self.base_model)
        #_average_gradients(self.top_model)
        def _average_loss(loss):
            size = float(dist.get_world_size())
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= size
            return loss
        loss = _average_loss(loss)
        return {'loss':loss}

    def _forward_val(self, data):
        input = data['img']
        label = data['label']
        output = self.base_model(input) 
        acc = self.top_model(output, label, mode='val')
        return {'acc': acc}

    def forward(self, data, mode='train'):
        if mode == 'train':
            start = time.time()
            re = self._forward_train(data)
            return re
        elif mode == 'val':
            return self._forward_puppy_val(data)
       
    def train(self, mode=True):
        self.base_model.train(mode)
        self.top_model.train(mode) 
