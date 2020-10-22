import torch
import cv2
import numpy as np
import os
import random
from ctypes import *
from torch.utils.data import Dataset
import mxnet as mx

import logging
import time
_format = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(format=_format, level='DEBUG')
_logger = logging.getLogger('debug')


class WebA1(Dataset):

    class PARAM(Structure):
        _fields_ = [
            ('label_width', c_int),
            ('mean_r', c_float),
            ('mean_g', c_float),
            ('mean_b', c_float),
            ('scale', c_float),
            ('height', c_int),
            ('width', c_int),
            ('channel', c_int),
            ('batch_size', c_int),
            ('resize_height', c_int),
            ('resize_width', c_int),
            ('patch_size', c_int),
            ('patch_idx', c_int),
            ('do_aug', c_bool),
            ('FacePatchSize_Main', c_int),
            ('FacePatchSize_Other', c_int),
            ('PatchFullSize', c_int),
            ('PatchCropSize', c_int),
            ('illum_trans_prob', c_float),
            ('gauss_blur_prob', c_float),
            ('motion_blur_prob', c_float),
            ('jpeg_comp_prob', c_float),
            ('res_change_prob', c_float),
            ('hsv_adjust_prob', c_float)
        ]

    lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_image_aug.so')
    lib = CDLL(lib_path, RTLD_GLOBAL)
    lib.FaceImageAugment.argtypes = [POINTER(PARAM), c_char_p]
    lib.FaceImageAugment.restype = POINTER(c_ubyte)
   
    def __init__(self,
                 rec_path,
                 transform_config,
                 test_mode=False):
        self.dataset = mx.io.ImageRecordIter(
              path_imgrec=rec_path, #'/mnt/data4/zcq/face/recognition/training/imgs/WebA1/val_WebA1_100w.rec',
              label_width=145,
              data_name='data',
              data_shape=(3, 400, 400),
              batch_size=1,
              shuffle=False,
              num_parts=1,
              part_index=0
          )


        start = time.time() 
        if test_mode:
            transform_config['do_aug'] = False
        self.transform_config = WebA1.PARAM(**transform_config)
        self.test_mode = test_mode
#        self.samples = []
#        with open(list_file) as f:
#            for line in f.readlines():
#                splits = line.strip().split(' ')
#                self.samples.append((os.path.join(data_root, splits[0]), int(splits[1])))
        _logger.info('dataset init, time used:{}'.format(time.time() - start))
        
 
    def __getitem__(self, index):
        #path, target = self.samples[index]
        #trans_img = self._transform(path)
        return {'img':trans_img, 'label':target}

    def _transform(self, path):
        return_data = WebA1.lib.FaceImageAugment(pointer(self.transform_config), path.encode()) 
        h = w = self.transform_config.PatchCropSize
        trans_img = np.ctypeslib.as_array(cast(return_data, POINTER(c_ubyte)), shape=(h, w, 3))
        trans_img = trans_img.astype(np.float32)
        trans_img = torch.from_numpy(trans_img.transpose((2, 0, 1)))
        return trans_img

    def __len__(self):
        return len(self.samples)
        
if __name__ == '__main__':
    transform_config = dict(
        label_width = 1,
        mean_r = 123.0,
        mean_g = 116.0,
        mean_b = 103.0,
        scale = 0.01,
        height = 108,
        width = 108,
        channel = 3,
        batch_size = 256,
        resize_height = 400,
        resize_width = 400,
        patch_size = 1,
        patch_idx = 0,
        do_aug = True,
        FacePatchSize_Main = 267,
        FacePatchSize_Other = 128,
        PatchFullSize = 128,
        PatchCropSize = 108,
        illum_trans_prob = 0.3,
        gauss_blur_prob = 0.3,
        motion_blur_prob = 0.1,
        jpeg_comp_prob = 0.4,
        res_change_prob = 0.4,
        hsv_adjust_prob = 0.1,)
    
    data_root = '.' #'/mnt/data4/huangchuanhong/datasets/puppy/recover_97w'

    dataset = WebA1(data_root, 'filter_short_train.lst', transform_config)
    print('len(dataset)=', len(dataset))
    for i in range(len(dataset)):
        trans_img, target = dataset[i]
        cv2.imwrite('debug/{}.jpg'.format(i), trans_img)
