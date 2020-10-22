import torch

#num_devices = len(allocator)
#num_batchs_per_epoch = int(num_images / batch_size / num_devices)

dist_config = dict(
  ip_rank_map={
    '35': 0
    #'245': 0,
    #'246': 1,
    #'244': 2,
  },
  master_addr='10.58.122.35',
  master_port='12346',
)

batch_size = 150
# model
base_model_gpus = [0]
top_model_gpus = [1]
model = dict(
  type='FaceModel',
  pretrained=None,
  base_model_gpus=base_model_gpus,
  top_model_gpus=top_model_gpus,
  base_model=dict(
    type='MobileFaceNetModel',
    feature_dim=192,
  ), 
  top_model=dict(
    type='AmSoftmax1GPUModel',
    feature_dim=192,
    num_classes=1000000,
    m=0.35,
    s=30.,
    bs_per_gpu=batch_size // len(top_model_gpus),
    gpus=top_model_gpus,
  )
)

data = dict(
  train_num_workers=32,
  val_num_workders=4,
  batch_size=batch_size,
  train_data = dict(
    type='Emore',
    data_root='/mnt/data4/huangchuanhong/datasets/faces_emore/imgs_tmp',
    ),
  val_data = dict(
    type='Emore',
    data_root='/mnt/data4/huangchuanhong/datasets/faces_emore/imgs_tmp',
    )
)

optimizer = dict(type='SGD', lr=0.15, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    step=[15, 20, 23, 25],#[13,16,19],
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=0.001)#1.0 / 3)
checkpoint_config = dict(interval=10000) # step, not epoch
val_config = dict(interval=10000000000) # step, not epoch
log_config = dict(interval=1, hooks=[dict(type='TextLoggerHook')]) # step, not epoch
total_epochs = 26
log_level = 'INFO'
work_dir = './work_dirs/emore_mobile_amsoftmax_new_tmp/'
load_from = None#'/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/softmax/latest.pth'
resume_from = None#'/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/softmax/latest.pth'

