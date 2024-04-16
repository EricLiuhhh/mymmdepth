_base_ = [
    '../_base_/datasets/kitti_completion.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepth'),
    dict(type='LoadCalibKitti'),
    dict(type='BottomCrop', crop_size=('auto', 375-100)),
    dict(type='RandomFlipBundle', prob=0.5),
    dict(type='RandomRotate', angle=5.0),
    dict(type='ColorJitter', 
        brightness=0.4,
        contrast=0.4,
        saturation=0.4),
    dict(type='RandomScaleBundle', scale=(1, 1.5)),
    dict(type='RandomCrop', crop_size=(1216, 240)),
    dict(type='Normalize', mean=[0, 0, 0], std=[255, 255, 255], to_rgb=False),
    dict(type='PackDepthInputs')
] 
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepth'),
    dict(type='BottomCrop', crop_size=('auto', 352-100)),
    dict(type='Normalize', mean=[0, 0, 0], std=[255, 255, 255], to_rgb=False),
    dict(type='PackDepthInputs')
] 
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))

data_preprocessor = dict(
    type='DepthDataPreprocessor',
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    bgr_to_rgb=True)

model = dict(
    type='NLSPNCompletor',
    data_preprocessor=data_preprocessor,
    encoder_decoder=dict(
        type='CFormerBackbone'
    ),
    refinement=dict(
        type='NLSPN',
        preserve_input=False,
        legacy=False,
        temperature=100,
        prop_time=6
    ),
    loss_cfg=dict(type='ComposeLoss', 
                loss_cfgs=[dict(type='NormLoss', p=1, min_depth=0.0, max_depth=90.0, normalize_type='sep_mean'),
                           dict(type='NormLoss', p=2, min_depth=0.0, max_depth=90.0, normalize_type='sep_mean')],
                loss_weights=[1.0, 1.0],
                loss_names=['l1loss', 'l2loss']
    )
)