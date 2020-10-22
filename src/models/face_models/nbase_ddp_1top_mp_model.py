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
class NBaseDDP1TopMPModel(nn.Module):
    def __init__(self, pretrained, base_model, top_model, base_model_gpus, base_model_ranks,
                 top_model_rank, batch_size, feature_dim):
        super(NBaseDDP1TopMPModel, self).__init__()
        self.base_model_ranks = base_model_ranks
        self.top_model_rank = top_model_rank
        self.rank = dist.get_rank()
        self.base_output_shape = (batch_size, feature_dim)
        self.base_model_ranks = base_model_ranks
        self.groups = {}
        for rank in base_model_ranks:
            self.groups[rank] = dist.new_group([self.top_model_rank, rank])
        self.base_model_group = dist.new_group(base_model_ranks)
        if self.rank == top_model_rank:
            # self.groups = []
            # for rank in base_model_ranks:
            #     self.groups.append(dist.new_group([self.top_model_rank, rank]))
            self.base_output_list = []
            self.label_list = []
            for rank in base_model_ranks:
                self.base_output_list.append(torch.zeros(self.base_output_shape, device='cuda:0', requires_grad=True, dtype=torch.float32))
                self.label_list.append(torch.zeros((batch_size,), device='cuda:0', dtype=torch.int64))
            self.top_model = build_top_model(top_model)
            print('{} top model= {}'.format(__file__, self.top_model))
            self.init_weights(pretrained)
            # self.top_model = nn.DataParallel(self.top_model, device_ids=top_model_gpus,
            #                                  output_device=top_model_gpus[0])
        else:
            # self.group =dist.new_group([self.top_model_rank, self.rank])
            # self.base_model_group = dist.new_group(base_model_ranks)
            self.output_grad = torch.zeros(self.base_output_shape, device=base_model_gpus[0], dtype=torch.float32)
            self.base_model = build_base_model(base_model).to('cuda:{}'.format(base_model_gpus[0]))
            self.init_weights(pretrained)
            print('{} base_model = {}'.format(__file__, self.base_model))
            self.base_model = DistributedDataParallel(self.base_model, device_ids=base_model_gpus,
                                                      output_device=base_model_gpus[0],
                                                      process_group=self.base_model_group)
            # self.base_model = nn.DataParallel(self.base_model, device_ids=base_model_gpus,
            #                                           output_device=base_model_gpus[0])

            # if len(self.base_model_ranks) > 1:
            #    for param in self.base_model.parameters():
            #        if param.requires_grad:
            #            param.register_hook(lambda grad:
            #                                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.base_model_group))
                 
    def init_weights(self, pretrained=None):
        if self.rank == self.top_model_rank:
            self.top_model.init_weights()
        else:
            self.base_model.init_weights(pretrained)

    def _forward_train(self, data):
        if self.rank == self.top_model_rank:
            for idx, rank in enumerate(self.base_model_ranks):
                if self.base_output_list[idx].grad is not None:
                    self.base_output_list[idx].grad *= 0
                dist.broadcast(tensor=self.base_output_list[idx], src=rank, group=self.groups[rank])
            for rank in self.base_model_ranks:
                dist.barrier(self.groups[rank])
            # print('receive base_output_list, time used ={}'.format(time.time() - start))
            # start = time.time()
            for idx, rank in enumerate(self.base_model_ranks):
                group = self.groups[rank]
                dist.broadcast(tensor=self.label_list[idx], src=rank, group=group)
            for rank in self.base_model_ranks:
                dist.barrier(self.groups[rank])
            # print('receive label_list, time used = {}'.format(time.time() - start))
            # start = time.time()
            total_base_output = torch.cat(self.base_output_list)
            total_label = torch.cat(self.label_list)
            # print('concat time used = {}'.format(time.time() - start))
            # start = time.time()
            loss = self.top_model(total_base_output, total_label).mean()
            # print('top forward time used = {}'.format(time.time() - start))
            # start = time.time()
            loss.backward()
            # print('loss backward time used = {}'.format(time.time() - start))
            # start = time.time()
            for idx, rank in enumerate(self.base_model_ranks):
                group = self.groups[rank]
                dist.broadcast(tensor=self.base_output_list[idx].grad, src=0, group=group)
            for idx, rank in enumerate(self.base_model_ranks):
                dist.barrier(self.groups[rank])
            # print('send base_output_list, time used ={}'.format(time.time() - start))
            return {'loss': loss}
        else:
            input = data['img'].to('cuda:0')
            label = data['label'].to('cuda:0')
            # start = time.time()
            output = self.base_model(input)
            # print('base_model forward, time used ={}'.format(time.time() - start))
            # start = time.time()
            dist.broadcast(tensor=output, src=self.rank, group=self.groups[self.rank])
            dist.barrier(group=self.groups[self.rank])
            # print('send output, time used ={}'.format(time.time() - start))
            # start = time.time()
            dist.broadcast(tensor=label, src=self.rank, group=self.groups[self.rank])
            dist.barrier(group=self.groups[self.rank])
            # print('send label, time used = {}'.format(time.time() - start))
            # start = time.time()
            dist.broadcast(tensor=self.output_grad, src=self.top_model_rank, group=self.groups[self.rank])
            dist.barrier(group=self.groups[self.rank])
            # print('receive output_grad, time used = {}'.format(time.time() - start))
            # if len(self.base_model_ranks) > 1:
            #    for param in self.base_model.parameters():
            #        if param.requires_grad:
            #            param.register_hook(lambda grad:
            #                                dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.base_model_group))
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
