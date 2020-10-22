from ctypes import *
import numpy as np
import cv2

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

lib = CDLL('./face_image_aug.so', RTLD_GLOBAL)
lib.FaceImageAugment.argtypes = [POINTER(PARAM), c_char_p]
lib.FaceImageAugment.restype = POINTER(c_ubyte)
param = PARAM(
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
        hsv_adjust_prob = 0.1,
)
print(param.resize_height)
img_file = "2.jpg".encode()
return_data = lib.FaceImageAugment(pointer(param), img_file)
x = np.ctypeslib.as_array(cast(return_data, POINTER(c_ubyte)), shape=(108, 108, 3))
print('x.shape=',x.shape)
cv2.imwrite('huang_x.jpg', x)

#class RESULT(Structure):
#    _fields_ = [("a", c_int), ("p", c_char_p)]
#lib = CDLL("./libfoo.so", RTLD_GLOBAL)
#add = lib.add
#add.argtypes = [c_int, c_int]
#add.restype = c_int
#a = 10
#b = 10
#c = add(a, b)
#print('c = {}'.format(c))
#test = lib.test
#test.argtypes = [c_char_p]
#test.restype = c_char_p
#sl = test(c_char_p("Hello".encode()))
#get_result = lib.get_result
#get_result.argtypes = [c_int, c_char_p]
#get_result.restype = POINTER(RESULT)
#a = 10
#my_str = "Hello".encode()
#ret = get_result(a, my_str)
#print("ret.a=%d, ret.p=%s"%(ret[0].a, ret[0].p))
