import torch.nn as nn
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from ...registry import FACE_MODEL
from ...builder import build_base_model, build_top_model

@FACE_MODEL.register_module
class DistPairwiseModel(nn.Module):
    def __init__(self, pretrained, base_model, top_model, base_model_gpus, top_model_gpu, batch_size, feature_dim):
        super(DistPairwiseModel, self).__init__()
        self.base_model = build_base_model(base_model).to('cuda:{}'.format(base_model_gpus[0]))
        self.top_model = build_top_model(top_model).to('cuda:{}'.format(top_model_gpu))
        self.init_weights(pretrained)
        # self.base_model = nn.DataParallel(self.base_model, device_ids=base_model_gpus, output_device=base_model_gpus[0])
        self.base_model = DistributedDataParallel(self.base_model, device_ids=base_model_gpus,
                                                  output_device=base_model_gpus[0])
        self.base_output_shape = (batch_size, feature_dim)
        self.base_output_list = []
        self.label_list = []
        for rank in range(dist.get_world_size()):
            if rank == dist.get_rank():
                self.base_output_list.append(None)
                self.label_list.append(None)
            else:
                self.base_output_list.append(
                    torch.zeros(self.base_output_shape, device='cuda:0', dtype=torch.float32))
                self.label_list.append(torch.zeros((batch_size,), device='cuda:0', dtype=torch.int64))
        self.rank = dist.get_rank()

    def init_weights(self, pretrained=None):
        self.base_model.init_weights(pretrained)
        self.top_model.init_weights()

    def _forward_train(self, data):
        input = data['img'].to('cuda:0')
        label = data['label'].to('cuda:0')
        output = self.base_model(input)
        self.base_output_list[self.rank] = output
        self.label_list[self.rank] = label
        for i in range(dist.get_world_size()):
            dist.broadcast(self.base_output_list[i], src=i)
            dist.barrier()
            dist.broadcast(self.label_list[i], src=i)
            dist.barrier()
        total_output = torch.cat(self.base_output_list, dim=0)
        total_label = torch.cat(self.label_list, dim=0)
        loss = self.top_model(total_output, total_label, mode='train')
        loss = loss.mean()
        loss.backward()
        return {'loss': loss}

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
