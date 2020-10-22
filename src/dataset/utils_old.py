import torch.distributed as dist
import torch
from random import Random
from ..utils import obj_from_dict
from . import dataset as dataset_factory

# class Partition(object):
#     def __init__(self, data, index):
#         self.data = data
#         self.index = index
#
#     def __len__(self):
#         return len(self.index)
#
#     def __getitem__(self, index):
#         data_idx = self.index[index]
#         return self.data[data_idx]
#
# class DataPartitioner(object):
#     def __init__(self, dataset, sizes=(0.5, 0.5), seed=0):
#         self.dataset = dataset
#         self.partitions = []
#         rng = Random()
#         rng.seed(seed)
#         data_len = len(dataset)
#         indexes = [x for x in range(0, data_len)]
#         rng.shuffle(indexes)
#         for frac in sizes:
#             part_len = int(frac * data_len)
#             self.partitions.append(indexes[0:part_len])
#             indexes = indexes[part_len:]
#
#     def use(self, partition):
#         return Partition(self.dataset, self.partitions[partition])
#
# class DataPrefetcher(object):
#     def __init__(self, loader):
#         self.data_len = len(loader)
#         self.loader = iter(loader)
#         self.stream = torch.cuda.Stream()
#         self.preload()
#
#     def preload(self):
#         try:
#             self.next_databatch = next(self.loader)
#         except StopIteration:
#             self.next_databatch = None
#         with torch.cuda.stream(self.stream):
#             input = self.next_databatch['img'].cuda(non_blocking=True)
#             label = self.next_databatch['label'].cuda(non_blocking=True)
#             self.next_databatch = {'img': input, 'label': label}
#
#     def __next__(self):
#         databatch = self.next_databatch
#         if databatch is None:
#            raise StopIteration
#         self.preload()
#         return databatch
#
#     def __iter__(self):
#         return self
#
#     def __len__(self):
#         return self.data_len
        
#
# def _dataloader_old(dataset, batch_size, num_workders, mode='train', prefetch=True):
#     world_size = dist.get_world_size()
#     partition_sizes = [1.0 / world_size for _ in range(world_size)]
#     partition = DataPartitioner(dataset, partition_sizes)
#     partition = partition.use(dist.get_rank())
#     if mode == 'train':
#         shuffle = True
#         drop_last = True
#     elif mode == 'val':
#         shuffle = False
#         drop_last = False
#     shuffle=False
#     data_loader = torch.utils.data.DataLoader(partition,
#                                               batch_size=batch_size,
#                                               shuffle=shuffle,
#                                               num_workers=num_workders,
#                                               drop_last=drop_last)
#     if prefetch:
#         data_loader = DataPrefetcher(data_loader)
#     return data_loader

def _dataloader(dataset, batch_size, num_workers, mode='train', nbase_1top=False):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if mode == 'train':
        drop_last = True
    elif mode == 'val':
        drop_last = False
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              drop_last=drop_last)
    return data_loader


def get_dataloader(data_config, num_workers, batch_size, mode='train'):
    dataset = obj_from_dict(data_config, dataset_factory) 
    dataloader = _dataloader(dataset, 
                             batch_size,
                             num_workers,
                             mode=mode)
    return dataloader

def get_onebase_1top_dataloader(data_config, num_workers, batch_size, mode='train'):
    dataset = obj_from_dict(data_config, dataset_factory)
    if mode == 'train':
        drop_last = True
        shuffle = True
    elif mode == 'val':
        drop_last = False
        shuffle = False
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last
    )
    return dataloader

def get_nbase_1top_dataloader(data_config, num_workers, batch_size, mode='train'):
    dataset = obj_from_dict(data_config, dataset_factory)
    if mode == 'train':
        drop_last = True
        shuffle = True
    elif mode == 'val':
        drop_last = False
        shuffle = False
    world_size = int(dist.get_world_size())
    rank = int(dist.get_rank())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size-1, rank=rank-1)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last
    )
    return dataloader