import torch.distributed as dist
import torch

def ndevice_dataloader(dataset, batch_size, num_workers, mode='train'):
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