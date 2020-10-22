import torch.nn as nn
import torch
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from ..registry import FACE_MODEL
from ..builder import build_base_model, build_top_model

@FACE_MODEL.register_module
class NBaseDDPMTopMPModel(nn.Module):
    def __init__(self, pretrained, base_model, top_model, base_model_gpus, top_model_gpus, base_model_ranks,
                 top_model_ranks, batch_size, feature_dim):
        assert(len(base_model_ranks) > 1)
        assert(len(top_model_ranks) > 1)
        super(NBaseDDPMTopMPModel, self).__init__()
        self.base_model_ranks = base_model_ranks
        self.top_model_ranks = top_model_ranks
        self.top_model_gpus = top_model_gpus
        self.rank = dist.get_rank()
        self.base_output_shape = (batch_size, feature_dim)
        # construct process groups
        # There are 4 kind of groups:
        # (1) top_group (2) base_tops_groups (3) top_base_groups (4) base_group
        self.top_group = dist.new_group(top_model_ranks)
        self.base_tops_groups = defaultdict(dict)
        for base_rank in base_model_ranks:
            group_idxes = [base_rank] + top_model_ranks
            self.base_tops_groups[base_rank] = dist.new_group(group_idxes)
        self.top_base_groups = defaultdict(dict)
        for top_rank in top_model_ranks:
            self.top_base_groups[top_rank] = dict()
            for base_rank in base_model_ranks:
                self.top_base_groups[top_rank][base_rank] = dist.new_group([top_rank, base_rank])
        self.base_group = dist.new_group(base_model_ranks)
        if self.rank in top_model_ranks:
            self.base_output_list = []
            self.label_list = []
            for rank in base_model_ranks:
                self.base_output_list.append(torch.zeros(self.base_output_shape, device=self.top_model_gpus[0], requires_grad=True, dtype=torch.float32))
                self.label_list.append(torch.zeros((batch_size,), device=self.top_model_gpus[0], dtype=torch.int64))
            top_model.top_group = self.top_group
            self.top_model = build_top_model(top_model)
            print('{} top model= {}'.format(__file__, self.top_model))
            self.init_weights(pretrained)
        else:
            self.output_grad_list = []
            for rank in self.top_model_ranks:
                self.output_grad_list.append(torch.zeros(self.base_output_shape, device=base_model_gpus[0], dtype=torch.float32))
            self.base_model = build_base_model(base_model).to('cuda:{}'.format(base_model_gpus[0]))
            self.init_weights(pretrained)
            print('{} base_model = {}'.format(__file__, self.base_model))
            self.base_model = DistributedDataParallel(self.base_model, device_ids=base_model_gpus,
                                                      output_device=base_model_gpus[0],
                                                      process_group=self.base_group)
                 
    def init_weights(self, pretrained=None):
        if self.rank in self.top_model_ranks:
            self.top_model.init_weights()
        else:
            self.base_model.init_weights(pretrained)

    def _forward_train(self, data):
        if self.rank in self.top_model_ranks:
            for idx, rank in enumerate(self.base_model_ranks):
                if self.base_output_list[idx].grad is not None:
                    self.base_output_list[idx].grad *= 0
                dist.broadcast(tensor=self.base_output_list[idx], src=rank, group=self.base_tops_groups[rank])
            for rank in self.base_model_ranks:
                dist.barrier(self.base_tops_groups[rank])
            for idx, rank in enumerate(self.base_model_ranks):
                dist.broadcast(tensor=self.label_list[idx], src=rank, group=self.base_tops_groups[rank])
            for rank in self.base_model_ranks:
                dist.barrier(self.base_tops_groups[rank])
            total_base_output = torch.cat(self.base_output_list)
            total_label = torch.cat(self.label_list)
            loss = self.top_model(total_base_output, total_label).mean()
            loss.backward()
            for idx, rank in enumerate(self.base_model_ranks):
                group = self.top_base_groups[self.rank][rank]
                dist.broadcast(tensor=self.base_output_list[idx].grad, src=self.rank, group=group)
            for idx, rank in enumerate(self.base_model_ranks):
                dist.barrier(self.top_base_groups[self.rank][rank])
            return {'loss': loss}
        else:
            input = data['img'].to('cuda:0')
            label = data['label'].to('cuda:0')
            output = self.base_model(input)
            dist.broadcast(tensor=output, src=self.rank, group=self.base_tops_groups[self.rank])
            dist.barrier(group=self.base_tops_groups[self.rank])
            dist.broadcast(tensor=label, src=self.rank, group=self.base_tops_groups[self.rank])
            dist.barrier(group=self.base_tops_groups[self.rank])
            for idx, rank in enumerate(self.top_model_ranks):
                dist.broadcast(tensor=self.output_grad_list[idx], src=rank, group=self.top_base_groups[rank][self.rank])
            for rank in self.top_model_ranks:
                dist.barrier(group=self.top_base_groups[rank][self.rank])
            output_grad = sum(self.output_grad_list)
            output.backward(output_grad)
            
    def _forward_val(self, data):
        input = data['img']
        label = data['label']
        output = self.base_model(input) 
        acc = self.top_model(output, label, mode='val')
        return {'acc': acc}

    def forward(self, data, mode='train'):
        if mode == 'train':
            re = self._forward_train(data)
            return re
        elif mode == 'val':
            return self._forward_puppy_val(data)
       
    def train(self, mode=True):
        if hasattr(self, 'base_model'):
            self.base_model.train(mode)
        elif hasattr(self, 'top_model'):
            self.top_model.train(mode) 
