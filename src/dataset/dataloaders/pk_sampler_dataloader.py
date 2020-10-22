import torch.distributed as dist
import torch
from ..samplers.balanced_batch_sampler import BalancedBatchSampler, DistributedBalancedBatchSampler

def pk_sampler_dataloader(dataset, n_classes, n_samples, num_workers):
    print('before_ balancedsampler')
    sampler = BalancedBatchSampler(dataset, n_classes, n_samples)
    print('after balanced_sampler')
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
    )
    return dataloader

def dist_pk_sampler_dataloader(dataset, n_classes, n_samples, num_workers, num_replicas=None, rank=None):
    sampler = DistributedBalancedBatchSampler(dataset, n_classes, n_samples, num_replicas, rank)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
    )
    return dataloader