import torch.distributed as dist
import torch

def nbase_1top_dataloader(dataset, batch_size, num_workers, mode='train'):
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