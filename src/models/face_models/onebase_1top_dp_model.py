import torch.nn as nn
import torch
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
class OneBase1TopDPModel(nn.Module):
    def __init__(self, pretrained, base_model, top_model, base_model_gpus, base_model_rank,
                 top_model_gpus, top_model_rank, batch_size, feature_dim):
        super(OneBase1TopDPModel, self).__init__()
        self.base_model_rank = base_model_rank
        self.top_model_rank = top_model_rank
        self.rank = dist.get_rank()
        self.base_output_shape = (batch_size, feature_dim)
        self.group = dist.new_group([self.top_model_rank, self.base_model_rank])
        if self.rank == top_model_rank:
            self.base_output = torch.zeros(self.base_output_shape, device='cuda:0', requires_grad=True, dtype=torch.float32)
            self.label = torch.zeros((batch_size,), device='cuda:0', dtype=torch.int64)
            self.top_model = build_top_model(top_model).to('cuda:{}'.format(top_model_gpus[0]))
            self.init_weights(pretrained)
            self.top_model = nn.DataParallel(self.top_model, device_ids=top_model_gpus,
                                             output_device=top_model_gpus[0])
        else:
            self.output_grad = torch.zeros(self.base_output_shape, device=base_model_gpus[0], dtype=torch.float32)
            print('{} before build_base_model'.format(__file__))
            self.base_model = build_base_model(base_model).to('cuda:{}'.format(base_model_gpus[0]))
            print('{} after build base model'.format(__file__))
            self.init_weights(pretrained)
            print('{} after init_weight'.format(__file__))
            print('{} base_model = {}'.format(__file__, self.base_model))
            # self.base_model = DistributedDataParallel(self.base_model, device_ids=base_model_gpus,
            #                                             output_device=base_model_gpus[0])
            self.base_model = nn.DataParallel(self.base_model, device_ids=base_model_gpus,
                                                      output_device=base_model_gpus[0])
            print('{} after distributeddataparallel'.format(__file__))
                 
    def init_weights(self, pretrained=None):
        if self.rank == self.top_model_rank:
            self.top_model.init_weights()
        else:
            self.base_model.init_weights(pretrained)

    def _forward_train(self, data=None):
        if self.rank == self.top_model_rank:
            if self.base_output.grad is not None:
                self.base_output.grad *= 0
            dist.broadcast(tensor=self.base_output, src=self.base_model_rank, group=self.group)
            dist.barrier(self.group)
            dist.broadcast(tensor=self.label, src=self.base_model_rank, group=self.group)
            dist.barrier(self.group)
            loss = self.top_model(self.base_output, self.label)
            loss = loss.mean()
            loss.backward()
            dist.broadcast(tensor=self.base_output.grad, src=self.top_model_rank, group=self.group)
            dist.barrier(self.group)
            return {'loss': loss}
        else:
            input = data['img'].to('cuda:0')
            label = data['label'].to('cuda:0')
            output = self.base_model(input)
            dist.broadcast(tensor=output, src=self.base_model_rank, group=self.group)
            dist.barrier(group=self.group)
            dist.broadcast(tensor=label, src=self.base_model_rank, group=self.group)
            dist.barrier(group=self.group)
            dist.broadcast(tensor=self.output_grad, src=self.top_model_rank, group=self.group)
            dist.barrier(group=self.group)
            output.backward(self.output_grad)
            
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
        if hasattr(self, 'base_model'):
            self.base_model.train(mode)
        elif hasattr(self, 'top_model'):
            self.top_model.train(mode) 
