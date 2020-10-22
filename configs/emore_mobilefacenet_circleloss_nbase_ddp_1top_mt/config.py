import torch

#num_devices = len(allocator)
#num_batchs_per_epoch = int(num_images / batch_size / num_devices)

dist_config = dict(
  ip_rank_map={
    '35': 0,
    '36': 1,
    #'244': 0,
    #'243': 1,
    #'248': 2,
    #'251': 3
  },
  master_addr='10.58.122.35',
  master_port='12345',
)

# model
base_model_gpus = [0,1]
base_model_ranks = [1]
top_model_rank = 0
batch_size = 256
feature_dim = 512
model = dict(
  type='NBaseDDP1TopMPModel',
  pretrained=None,#'modelzoo://resnet50',
  base_model_gpus=base_model_gpus,
  base_model_ranks=base_model_ranks,
  top_model_rank=top_model_rank,
  batch_size=batch_size,
  feature_dim=feature_dim,
  base_model=dict(
    #type='ResNext50Model',
    type='MobileFaceNetModel',
    feature_dim=feature_dim,
  ),
  top_model=dict(
    type='CircleLossMP',
    feature_dim=feature_dim,
    #num_classes=85742,#967410,
    #num_classes=967410,
    num_classes=10001,
    m=0.25,
    gamma=256.,
  )
)

data = dict(
  train_num_workers=32,
  val_num_workders=4,
  batch_size=batch_size,
  train_data=dict(
    type='Emore',
    data_root='/mnt/data4/huangchuanhong/datasets/faces_emore/imgs_tmp',
    ),
  val_data=dict(
    type='Emore',
    data_root='/mnt/data4/huangchuanhong/datasets/faces_emore/imgs_tmp',
    )
)

optimizer = dict(type='SGD', lr=0.1 * len(base_model_ranks), momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    step=[5,8,10],
    warmup='linear',
    warmup_iters=400,
    warmup_ratio=0.001)#1.0 / 3)
checkpoint_config = dict(interval=2000000) # step, not epoch
val_config = dict(interval=10000000000) # step, not epoch
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')]) # step, not epoch
total_epochs = 12
log_level = 'INFO'
#work_dir = '/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/97w_resfacenext_amsoftmax'
work_dir = '/mnt/data1/huangchuanhong/checkpoints/dist_face_pytorch/work_dirs/mobilefacenet_nbase_ddp_1top_mt_c10001_circleloss'
load_from = None#'/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/softmax/latest.pth'
resume_from = None#'/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/softmax/latest.pth'


