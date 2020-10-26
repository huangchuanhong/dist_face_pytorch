# Introduction
## Description
Distributed train framework for face-recgnition with large IDs(number of classes). Implemented with Pytorch
## loss type
am-softmax or circleloss
## framework
### framework1: Nbase_1top
N devices(servers) are used as base model(backbone) devices (data parallel), <br/>
one device(server) as top model(the last feature_dim x num_classes fc and loss) device (model parallel).
### framework2: Nbase_Mtop
N devices(servers) are used as base model(backbone) devices (data parallel), <br/>
M device(server) as top model(the last feature_dim x num_classes fc and loss) device (model parallel).
# train
## Nbase_1top
```bash
python tools/nbase_1top_mp_train.py configs/97w_ecaregnet64_circloeloss_nbase_ddp_1top_mt/config.py
```
## Nbase_mtop
```bash
python tools/nbase_mtop_mp_train.py configs/97w_ecaregnet64_192_circleloss_nbase_ddp_mtop/config.py
```
# evaluate
```bash
# transform origin saved model
python tools/public_val/trans_checkpoints.py --input_path=/where/the/saved/model/locate/latest.pth
# evaluate
python tools/public_val/val.py
```
# convert to caffe
```bash
python tools/torch_to_caffe/torch2caffe_classify.py config_path checkpoint_path name
```
