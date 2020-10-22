import torch

#num_devices = len(allocator)
#num_batchs_per_epoch = int(num_images / batch_size / num_devices)

dist_config = dict(
  ip_rank_map={
    '249': 0,
    '251': 1,
    '243': 2,
    '244': 3,
    '245': 4,
    '246': 5,
    '248': 6,
  },
  master_addr='172.24.1.249',
  master_port='12345',
)


# model
base_model_gpus = [0,1,2,3,4,5,6]
base_model_ranks = [2,3,4,5,6]
top_model_gpus = [0,1,2,3,4,5,6]
top_model_ranks = [0,1]
batch_size = 512 #736#640
feature_dim = 192
model = dict(
  type='NBaseDDPMTopMPModel',
  pretrained=None,  # 'modelzoo://resnet50',
  base_model_gpus=base_model_gpus,
  top_model_gpus=top_model_gpus,
  base_model_ranks=base_model_ranks,
  top_model_ranks=top_model_ranks,
  batch_size=batch_size,
  feature_dim=feature_dim,
  base_model=dict(
    type='RegnetModel',
    #type='RegnetModel',
    feature_dim=feature_dim,
    gflops='3.2GF',
    num_stages=4,
    out_indices=(3,),
  ),
  top_model=dict(
    type='CircleLossNTopMP',
    batch_size=batch_size * len(base_model_ranks),
    feature_dim=feature_dim,
    # num_classes=85742,#967410,
    # num_classes=967410,
    top_model_ranks=top_model_ranks,
    top_model_gpus=top_model_gpus,
    num_classes=967410,
    m=0.25,
    s=256.,
  )
)

data = dict(
  train_num_workers=32,
  val_num_workders=4,
  batch_size=batch_size,
  train_data=dict(
    type='WebA1ConcateLmdb',
    data_root_list = ['/mnt/data{}/huangchuanhong/datasets/weba1_splits_lmdb_{}'.format(i+1, i) for i in range(4)],
    #data_root_list = ['/mnt/data{}/zhaoyu/25w_id_{}'.format(i, i) for i in range(1, 5)], 
  ),
  train_dataloader=dict(
    type='nbase_mtop_dataloader',
    batch_size=batch_size,
    top_count=len(top_model_ranks),
    num_workers=32,
    mode='train'
  ),
  val_data=dict(
    type='WebA1ConcateLmdb',
    data_root_list=['/mnt/data{}/huangchuanhong/datasets/weba1_splits_lmdb_{}'.format(i + 1, i) for i in range(4)],
    #data_root_list = ['/mnt/data{}/zhaoyu/25w_id_{}'.format(i, i) for i in range(1, 5)],
  )
)

optimizer = dict(type='SGD', lr=0.1 * len(base_model_ranks), momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    step=[7,11,13],
    warmup='linear',
    warmup_iters=23689,#39482,
    warmup_ratio=0.0001)#1.0 / 3)
checkpoint_config = dict(interval=5000) # step, not epoch
val_config = dict(interval=10000000000) # step, not epoch
log_config = dict(interval=20, hooks=[dict(type='TextLoggerHook')]) # step, not epoch
total_epochs = 14
log_level = 'INFO'
#work_dir = '/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/97w_resfacenext_amsoftmax'
work_dir = '/mnt/data1/huangchuanhong/checkpoints/dist_face_pytorch/work_dirs/97w_regnet_circleloss_nbase_mtop_243_244_245_246_248_249_251'
load_from = None#'/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/softmax/latest.pth'
resume_from = None#'/mnt/data4/huangchuanhong/code/dist_face_pytorch/work_dirs/softmax/latest.pth'


