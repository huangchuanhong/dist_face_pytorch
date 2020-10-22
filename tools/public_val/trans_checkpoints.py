import argparse
import torch
import sys
import numpy as np
sys.path.insert(0, 'tools/public_val')
from srcs.data import get_val_data
from srcs.evaluate import val
from srcs.utils import l2_norm
from srcs.checkpoint import load_state_dict
sys.path.insert(0, '.')
from src.config import Config
from src.models import build_base_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', default='tools/public_val/latest.pth', help='input ckpt path')
    parser.add_argument('--output_path', default='tools/public_val/ckpt.pth', help='output ckpt path') 
    return parser.parse_args()
     
def main():
    args = parse_args()
    state_dict = {'state_dict': torch.load(args.input_path)['state_dict']}
    torch.save(state_dict, args.output_path)

if __name__ == '__main__':
    main()
