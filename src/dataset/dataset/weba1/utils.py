import torch.distributed as dist
import torch
from random import Random
from .weba1 import WebA1

class Partition(object):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):
    def __init__(self, dataset, sizes=(0.5, 0.5), seed=0):
        self.dataset = dataset
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(dataset)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)
        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.dataset, self.partitions[partition])

def _dataloader(dataset, batch_size, num_workders, mode='train'):
    world_size = dist.get_world_size()
    partition_sizes = [1.0 / world_size for _ in range(world_size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    if mode == 'train':
        shuffle = True
        drop_last = True
    elif mode == 'val':
        shuffle = False
        drop_last = False
    data_loader = torch.utils.data.DataLoader(partition,
                                              batch_size=batch_size, 
                                              shuffle=shuffle, 
                                              num_workers=num_workders, 
                                              drop_last=drop_last)
    return data_loader 

def get_dataloader(data_config, num_workers, mode='train'):
    dataset = WebA1(**data_config)
    dataloader = _dataloader(dataset, 
                             data_config.transform_config.batch_size,
                             num_workers,
                             mode=mode)
    return dataloader
