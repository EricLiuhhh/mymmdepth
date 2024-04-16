dataset_type = 'NuScenesCompletionRD'
data_root = 'data/nuscenes_depth_rd'
backend_args = None
input_shape = (800, 450)
train_pipeline = [
    dict(type='LoadH5', key_map={
            'image': 'img',
            'lidar_depth': 'gt_depth',
            'radar_depth': 'sparse_depth'
        }),
    dict(type='ConvertColor', flag='rgb2bgr'),
    dict(type='DepthRangeFilter', min_depth=0.0, max_depth=80.0, keys=['sparse_depth']),
    dict(type='RandomRotate', angle=5.0),
    dict(type='RandomScaleBundle', scale=(1, 1.5)),
    dict(type='RandomCrop', crop_size=input_shape),
    dict(type='RandomFlipBundle', prob=0.5),
    dict(type='ColorJitter', 
        brightness=0.2,
        contrast=0.2,
        saturation=0.2),
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
        ann_file=f'nuscenes_completion_rd_infos_train.pkl',
        data_prefix={},
        pipeline=train_pipeline,
        test_mode=False)
)
val_pipeline = [
    dict(type='LoadH5', key_map={
            'image': 'img',
            'lidar_depth': 'gt_depth',
            'radar_depth': 'sparse_depth'
        }),
    dict(type='ConvertColor', flag='rgb2bgr'),
    dict(type='DepthRangeFilter', min_depth=0.0, max_depth=80.0, keys=['sparse_depth']),
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
        ann_file=f'nuscenes_completion_rd_infos_val.pkl',
        data_prefix={},
        pipeline=val_pipeline,
        test_mode=True)
)
val_evaluator = dict(type='KittiMetric')