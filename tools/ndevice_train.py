# Note: Can't not work properly. Don't use this
import argparse
import sys
import torch
import torch.distributed as dist
sys.path.insert(0, '.')
from src.config import Config
from src.models import build_model
from src.dataset import get_dataloader
from src.apis import init_process, train_model
from src.runner.utils import init_logger

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work_dir', help='the dir to save logs and models')
    args = parser.parse_args()
    return args

def main():
    torch.manual_seed(0)
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    _logger = init_logger(cfg.work_dir, 'INFO')
    _logger.info(cfg)

    
    print('before init_process')
    init_process(cfg.dist_config)

    rank = dist.get_rank()
    print('rank={}'.format(rank))
    print('world_size={}'.format(dist.get_world_size()))

    print('after init_process')
    print('before build_model')
    model = build_model(cfg.model)
    print('after build_model')
    print('before train_dataloader')

    train_dataloader = get_dataloader(cfg.data.train_data, cfg.data.train_dataloader)
    val_dataloader = train_dataloader
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    cfg.data.dataloader_lens = cfg.data.train_num_samples // cfg.data.batch_size
    #cfg.data.dataloader_lens = 682240 // len(cfg.base_model_ranks) // cfg.data.batch_size
    #cfg.data.dataloader_lens =  60644986 // len(cfg.base_model_ranks) // cfg.data.batch_size
    #cfg.data.dataloader_lens = 15654036 // len(cfg.base_model_ranks) // cfg.data.batch_size
    try:
        train_model(
          model, 
          dataloaders,
          cfg,
        )     
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
        dist.destroy_process_group()
 
if __name__ == '__main__':
    main()
