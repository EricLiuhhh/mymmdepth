dataset_type = 'KittiCompletion'
data_root = 'data/kitti_depth_esti'
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepth'),
    dict(type='BottomCrop', crop_size=(1216, 352)),
    dict(type='ColorJitter', 
        brightness=0.4,
        contrast=0.4,
        saturation=0.4),
    dict(type='RandomFlipBundle', prob=0.5),
    dict(type='RandomRotate', angle=5.0),
    dict(type='RandomScaleBundle', scale=(1, 1.5)),
    dict(type='RandomCrop', crop_size=(1216, 256)),
    dict(type='PackDepthInputs')
]    
train_dataloader = dict(
    batch_size=2,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_completion_infos_eigen_zhou_train.pkl',
        data_prefix={},
        pipeline=train_pipeline,
        test_mode=False)
)
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepth'),
    dict(type='PackDepthInputs')
]    
val_dataloader = dict(
    batch_size=1,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        #indices=list(range(100)),
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_completion_infos_eigen_test.pkl',
        data_prefix={},
        pipeline=val_pipeline,
        test_mode=True)
)
#val_evaluator = dict(type='KittiMetric', eval_crop=(153, 371, 44, 1197), scale_recovery=True, depth_clamp=(1e-3, 80))

val_evaluator = dict(type='KittiMetric', eval_crop=(0.40810811, 0.99189189, 0.03594771, 0.96405229), scale_recovery=True, depth_clamp=(1e-3, 80))
# val_evaluator = dict(type='KittiMetric', scale_recovery=True, depth_clamp=(1e-3, 80))