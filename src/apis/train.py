import socket
from collections import OrderedDict
import os
import torch
import torch.distributed as dist
from ..runner import Runner
import torch.distributed as dist

def parse_losses(losses):
    loss_vars = OrderedDict()
    summary_vars = OrderedDict()
    for loss_name, loss_value in losses.items():
        #if not 'loss' in loss_name:
        #    summary_vars[loss_name] = loss_value
        #    continue
        if isinstance(loss_value, torch.Tensor):
            loss_vars[loss_name] = loss_value.mean()
        elif isinstance(loss_value, list):
            loss_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
        else:
            raise TypeError(
                '{} is not a tensor or list of tensors'.format(loss_name))
        summary_vars[loss_name] = loss_value

    loss = sum(_value for _key, _value in loss_vars.items() if 'loss' in _key)

    loss_vars['loss'] = loss
    for name in loss_vars:
        loss_vars[name] = loss_vars[name].item()
    for name in summary_vars:
        summary_vars[name] = summary_vars[name].detach().cpu()

    return loss, loss_vars, summary_vars

def batch_processor(model, data, mode='train', num_samples=80):
    if mode == 'train':
        outputs = model(data, mode='train')
        loss, log_vars, summary_vars = parse_losses(outputs)
        if data:
            outputs = dict(
                loss=loss, log_vars=log_vars, summary_vars=summary_vars, num_samples=len(data['img'].data))
        else:
            outputs = dict(
                loss=loss, log_vars=log_vars, summary_vars=summary_vars, num_samples=num_samples
            )
    elif mode == 'val':
        acc_d = model(data, mode='val')
        outputs = dict(log_vars=acc_d, num_samples=len(data['img'].data))
    return outputs

# def batch_processor(model, data, mode='train', num_samples=80):
#     if mode == 'train':
#         losses = model(data, mode='train')
#         loss, log_vars = parse_losses(losses)
#         if data:
#             outputs = dict(
#                 loss=loss, log_vars=log_vars, num_samples=len(data['img'].data))
#         else:
#             outputs = dict(
#                 loss = loss, log_vars=log_vars, num_samples=num_samples
#             )
#     elif mode == 'val':
#         acc_d = model(data, mode='val')
#         outputs = dict(log_vars=acc_d, num_samples=len(data['img'].data))
#     return outputs


def get_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip

def init_process(cfg): 
    backend = 'nccl'
    ip = get_ip().split('.')[-1]
    rank = cfg.ip_rank_map[ip]
    size = len(cfg.ip_rank_map)
    os.environ['MASTER_ADDR'] = cfg.master_addr
    os.environ['MASTER_PORT'] = cfg.master_port
    os.environ['WORLD_SIZE'] = str(size)
    os.environ['RNAK'] = str(rank)
    dist.init_process_group(backend, rank=rank, world_size=size)

def train_model(
      model,
      dataloaders,
      cfg,
    ):
    assert(isinstance(dataloaders, dict))
    if 'train' not in dataloaders:
        dataloaders['train'] = None
    if 'val' not in dataloaders:
        dataloaders['val'] = None
    runner = Runner(
        model, 
        batch_processor,
        cfg.optimizer,
        cfg.work_dir,
        cfg.log_level, 
    )
    runner.register_training_hooks(
         cfg.lr_config,
         cfg.optimizer_config,
         cfg.checkpoint_config,
         cfg.val_config,
         cfg.log_config,
         #cfg.summary_config,
    )
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(cfg.total_epochs, dataloaders)

def train_nbase_1top_model(
        model,
        dataloaders,
        cfg
):
    rank = dist.get_rank()
    if rank != cfg.top_model_rank:
        cfg.optimizer['lr'] *= len(cfg.base_model_ranks)
        print('updating lr = {}'.format(cfg.optimizer['lr']))
        #cfg.optimizer['weight_decay'] /= len(cfg.base_model_ranks)
        #print('updating weight_decay = {}'.format(cfg.optimizer['weight_decay']))
    #if rank == cfg.top_model_rank:
    #    cfg.optimizer['weight_decay'] = 0.
    #    print('updating top weight_decay = 0')
    runner = Runner(
        model,
        batch_processor,
        cfg.optimizer,
        cfg.work_dir,
        cfg.log_level,
    )
    runner.register_training_hooks(
        cfg.lr_config,
        cfg.optimizer_config,
        cfg.checkpoint_config,
        cfg.val_config,
        cfg.log_config,
        #cfg.summary_config,
    )
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.nbase_1top_run(cfg.total_epochs, dataloaders, cfg.data.dataloader_lens, cfg.data.batch_size)

def train_nbase_mtop_model(
        model,
        dataloaders,
        cfg
):
    rank = dist.get_rank()
    if rank not in cfg.top_model_ranks:
        cfg.optimizer['lr'] *= len(cfg.base_model_ranks)
        print('updating lr = {}'.format(cfg.optimizer['lr']))
        cfg.optimizer['weight_decay'] = 2e-4 / len(cfg.base_model_ranks)
        print('updating weight_decay = {}'.format(cfg.optimizer['weight_decay']))
    #if rank == cfg.top_model_rank:
    #    cfg.optimizer['weight_decay'] = 0.
    #    print('updating top weight_decay = 0')
    runner = Runner(
        model,
        batch_processor,
        cfg.optimizer,
        cfg.work_dir,
        cfg.log_level,
    )
    runner.register_training_hooks(
        cfg.lr_config,
        cfg.optimizer_config,
        cfg.checkpoint_config,
        cfg.val_config,
        cfg.log_config,
        #cfg.summary_config,
    )
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.nbase_mtop_run(cfg.total_epochs, dataloaders, cfg.data.dataloader_lens, cfg.data.batch_size,
                          cfg.top_model_ranks)

                                                

