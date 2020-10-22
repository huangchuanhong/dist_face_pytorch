# Introduction
## Description
Distributed train framework for face-recgnition with large IDs(number of classes). Implemented with Pytorch 
## loss type
Only Am-softmax has been implemented now
## framework
N devices(servers) are used as base model(backbone) devices, <br/>
one device(server) as top model(the last feature_dim x num_classes fc and amsoftmax loss) device.
## hardware
what we have are: <br/>
4 servers each with 7 1080ti GPUs. <br/>
3 for base model, 1 for top model
# train
```base
python tools/nbase_1top_mp_train.py configs/97w_regnet_amsoftmax_nbase_ddp_1top_mt/config.py
```
# evaluate
```bash
# transform origin saved model
python tools/public_val/trans_checkpoints.py --input_path=/where/the/saved/model/locate/latest.pth
# evaluate
python tools/public_val/val.py
```