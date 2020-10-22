import argparse
import os
import torch
from torch.autograd import Variable
import sys
import cv2
import numpy as np
sys.path.insert(0, 'tools/torch_to_caffe')
from srcs.checkpoint import load_state_dict
from srcs.nn_tools import pytorch_to_caffe
sys.path.insert(0, '.')
from src.config import Config
from src.models import build_base_model
torch.set_printoptions(precision=8)

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('name', help='save model name')
    parser.add_argument('--save_dir', help='the prototxt and caffemodel save directory', default='.')
    parser.add_argument('--test_img', help='do test with demo input', default='tools/torch_to_caffe/img.jpg')
    args = parser.parse_args()
    return args

def preprocess(img_file):
    mean = np.array([0.482352, 0.45490, 0.40392])
    std = np.array([0.392157, 0.392157, 0.392157])
    img = cv2.imread(img_file)
    img = img[..., ::-1]
    img = cv2.resize(img, (108, 108))
    img = (img / 255. - mean) / std
    img = img.transpose((2, 0, 1))[np.newaxis, ...]
    return img

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    model = build_base_model(cfg.model.base_model)
    state_dict = torch.load(args.checkpoint,map_location='cpu')['state_dict']
    load_state_dict(model, state_dict)
    model.eval()
    if args.test_img:
        input = torch.tensor(preprocess(args.test_img), dtype=torch.float32)
        input = Variable(input)
    else:
        input=Variable(torch.ones([1, 3, 108, 108]))
    out = pytorch_to_caffe.trans_net(model, input, args.name)
    print(out)
    pytorch_to_caffe.save_prototxt(os.path.join(args.save_dir, '{}.prototxt'.format(args.name)))
    pytorch_to_caffe.save_caffemodel(os.path.join('{}.caffemodel'.format(args.name)))

if __name__ == '__main__':
    main()
