import torch

#num_devices = len(allocator)
#num_batchs_per_epoch = int(num_images / batch_size / num_devices)

dist_config = dict(
  ip_rank_map={
    '246': 0,
    '245': 1,
  },
  master_addr='172.24.1.246',
  master_port='12345',
)


# model
base_model_gpus = [0,1]
top_model_gpu = 0
n_classes = 4
n_samples = 4
batch_size = n_classes * n_samples
feature_dim = 192
model = dict(
  type='DistPairwiseModel',
  pretrained=None,#'modelzoo://resnet50',
  base_model_gpus=base_model_gpus,
  top_model_gpu=top_model_gpu,
  batch_size=batch_size,
  feature_dim=feature_dim,
  base_model=dict(
    #type='ResNext50Model',
    type='MobileFaceNetModel',
    feature_dim=feature_dim,
  ),
  top_model=dict(
    type='PairwiseCirclelossModel',
    m=0.25,
    gamma=256.,
    gpu=0,
  )
)

data = dict(
  train_data=dict(
    type='Emore',
    data_root='/mnt/data4/huangchuanhong/datasets/faces_emore/imgs_tmp',
    ),
  train_dataloader=dict(
    type='dist_pk_sampler_dataloader',
    n_classes=n_classes,
    n_samples=n_samples,
    num_workers=32
  ),
  val_data = dict(
    type='Emore',
    data_root='/mnt/data4/huangchuanhong/datasets/faces_emore/imgs_tmp',
    )
)


optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    step=[5,8,10],
    warmup='linear',
    warmup_iters=800,
    warmup_ratio=0.001)#1.0 / 3)
checkpoint_config = dict(interval=2000) # step, not epoch
val_config = dict(interval=10000000000) # step, not epoch
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')]) # step, not epoch
total_epochs = 12
log_level = 'INFO'
work_dir = '/mnt/data1/huangchuanhong/checkpoints/dist_face_pytorch/work_dirs/circleloss_dist'
load_from = None
resume_from = None


