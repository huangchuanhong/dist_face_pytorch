import torch

#num_devices = len(allocator)
#num_batchs_per_epoch = int(num_images / batch_size / num_devices)

dist_config = dict(
  ip_rank_map={
    '35': 0
  },
  master_addr='10.58.122.35',
  master_port='12345',
)


# model
base_model_gpus = [0]#,1,2,3,4,5]
top_model_gpus = [1]
batch_size = 64 * len(base_model_gpus) #512 #736#640
feature_dim = 192
model = dict(
  type='FaceModel',
  pretrained=None,#'modelzoo://resnet50',
  base_model_gpus=base_model_gpus,
  top_model_gpus=top_model_gpus,
  base_model=dict(
    type='DLAModel',
    feature_dim=feature_dim,
    last_channels=1024,
    levels=[1, 3, 4, 1],
    channels=[32, 128, 256, 512, 1024],
    block='BottleneckX'
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
  batch_size=batch_size,
  train_num_samples=5822653,
  train_data = dict(
    type='Emore',
    data_root='/mnt/data4/huangchuanhong/datasets/faces_emore/imgs_tmp',
    ),
  val_data = dict(
    type='Emore',
    data_root='/mnt/data4/huangchuanhong/datasets/faces_emore/imgs_tmp',
    ),
  # train_data=dict(
  #   type='ImgList',
  #   data_root='/mnt/data1/zhaoyu/faces_emore_p1',
  #   list_file='/mnt/data1/zhaoyu/faces_emore_p1/imglabellist',
  #   mean=[0.482352, 0.45490, 0.40392],
  #   std=[0.392157, 0.392157, 0.392157],
  #   random_crop=True,
  #   ),
  train_dataloader=dict(
    type='ndevice_dataloader',
    batch_size=batch_size,
    num_workers=32,
    mode='train'
  ),
  # val_data=dict(
  #   type='Emore',
  #   data_root='/mnt/data4/huangchuanhong/datasets/faces_emore/imgs',
  #   )
)

optimizer = dict(type='SGD', lr=0.1 * batch_size / 512., momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    step=[6,10,12,13],
    warmup='linear',
    warmup_iters=10000,#39482,
    warmup_ratio=0.0005)#1.0 / 3)
checkpoint_config = dict(interval=100) # step, not epoch
val_config = dict(interval=10000000000) # step, not epoch
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')]) # step, not epoch
total_epochs = 14
log_level = 'INFO'
#work_dir = '/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/97w_resfacenext_amsoftmax'
work_dir = '/mnt/data1/huangchuanhong/checkpoints/dist_face_pytorch/work_dirs/emore_dla_amsoftmax_ndevice_36'
load_from = None#'/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/softmax/latest.pth'
resume_from = None#'/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/softmax/latest.pth'


