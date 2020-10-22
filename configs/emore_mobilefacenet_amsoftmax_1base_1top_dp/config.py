import torch

#num_devices = len(allocator)
#num_batchs_per_epoch = int(num_images / batch_size / num_devices)

dist_config = dict(
  ip_rank_map={
    '241': 1,
    '244': 0,
    #'241': 3,
  },
  master_addr='172.24.1.244',
  master_port='12345',
)


# model
base_model_gpus = [0,1,2,3,4,5,6]
base_model_rank = 1
top_model_gpus = [0,1,2,3,4,5,6]
top_model_rank = 0
batch_size = 770
feature_dim = 512
model = dict(
  type='OneBase1TopDPModel',
  pretrained=None,#'modelzoo://resnet50',
  base_model_gpus=base_model_gpus,
  base_model_rank=base_model_rank,
  top_model_gpus=top_model_gpus,
  top_model_rank=top_model_rank,
  batch_size=batch_size,
  feature_dim=feature_dim,
  base_model=dict(
    #type='ResNext50Model',
    type='MobileFaceNetModel',
    feature_dim=feature_dim,
  ),
  top_model=dict(
    type='AmSoftmax1GPUModel',
    feature_dim=feature_dim,
    num_classes=85742,#967410,
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


optimizer = dict(type='SGD', lr=1.0, momentum=0.9, weight_decay=0.0001)
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
#work_dir = '/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/97w_resfacenext_amsoftmax'
work_dir = '/mnt/data1/huangchuanhong/checkpoints/dist_face_pytorch/work_dirs/mobilefacenet_1base1top_dp'
load_from = None#'/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/softmax/latest.pth'
resume_from = None#'/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/softmax/latest.pth'


