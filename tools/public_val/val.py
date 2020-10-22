import argparse
import torch
import sys
import numpy as np
sys.path.insert(0, 'tools/public_val')
from srcs.data import get_val_data
from srcs.evaluate import val
from srcs.utils import l2_norm, gen_plot
from srcs.checkpoint import load_state_dict
sys.path.insert(0, '.')
from src.config import Config
from src.models import build_base_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')#, default='configs/zy_emore_regnet/config.py', help='config file path')
    parser.add_argument('ckpt_path')#, default='tools/public_val/ckpt.pth', help='the dir to save logs and models')
    parser.add_argument('--data_root', default='/mnt/data4/huangchuanhong/datasets/faces_emore', help='root dir of the val data')
    parser.add_argument('--our_norm', action='store_true', default=False)
    return parser.parse_args()

     
def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.batch_size = 1000
    agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame = get_val_data(args.data_root)
    model = build_base_model(cfg.model.base_model).to('cuda')
    state_dict = torch.load(args.ckpt_path)['state_dict']
    load_state_dict(model, state_dict)
    agedb_acc, agedb_best_thresh, tpr, fpr = val(cfg, model, agedb_30, agedb_30_issame, our_normalization=args.our_norm)
    gen_plot(fpr, tpr, 'tools/public_val/agedb_roc_curve.jpg')
    print('agedb_acc={}, agedb_best_thresh={}'.format(agedb_acc, agedb_best_thresh))
    cfp_acc, cfp_best_thresh, tpr, fpr = val(cfg, model, cfp_fp, cfp_fp_issame, our_normalization=args.our_norm)
    gen_plot(fpr, tpr, 'tools/public_val/cfg_roc_curve.jpg')
    print('cfp_fp_acc={}, cfp_best_thresh={}'.format(cfp_acc, cfp_best_thresh))
    lfw_acc, lfw_best_thresh, tpr, fpr = val(cfg, model, lfw, lfw_issame, our_normalization=args.our_norm)
    gen_plot(fpr, tpr, 'tools/public_val/lfw_roc_curve.jpg')
    print('lfw_acc={}, lfw_best_thresh={}'.format(lfw_acc, lfw_best_thresh))

if __name__ == '__main__':
    main()
