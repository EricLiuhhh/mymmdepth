_base_ = [
    '../_base_/datasets/kitti_completion.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
backend_args = None

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=45, val_interval=2)
# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=1, convert_to_iter_based=True),
    dict(
        type='MultiStepLR',
        begin=1,
        end=45,
        by_epoch=True,
        milestones=[20, 25, 30, 35, 40, 45],
        gamma=0.5)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.05))

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
    #dict(type='RandomScaleBundle', scale=(1, 1.5)),
    dict(type='RandomCrop', crop_size=(1216, 256)),
    dict(type='FilterDepth'),
    dict(type='PreFillDepth'),
    dict(type='PackDepthInputs')
] 
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepth'),
    dict(type='FilterDepth'),
    dict(type='PreFillDepth'),
    dict(type='PackDepthInputs')
] 
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))

data_preprocessor = dict(
    type='DepthDataPreprocessor',
    mean=(90.995, 96.2278, 94.3213),
    std=(79.2382, 80.5267, 82.1483),
    bgr_to_rgb=True)

model = dict(
    type='LRRU',
    data_preprocessor=data_preprocessor,
    img_encoder=dict(
        type='StoDepthResNet',
        in_channels=3,
        base_channels=32,
        num_blocks=(2, 2, 2, 2, 2),
        plane_ratios=(2, 4, 8, 8, 8),
        strides=(1, 2, 2, 2, 2),
        hook_positions=dict(l=(1, 2, 3, 4, 5))
    ),
    depth_guide=dict(
        guide_cfg=dict(type='CatConvGuide'),
        guide_map=tuple(zip(('l1', 'l2', 'l3', 'l4'), ('l1', 'l2', 'l3', 'l4'))),
    ),
    decode_guide=dict(
        guide_cfg=dict(type='AddGuide'),
        guide_map=tuple(zip(('g3', 'g2', 'g1', 'g0'), ('l0', 'l1', 'l2', 'l3')))
    ),
    depth_encoder=dict(
        type='StoDepthResNet',
        in_channels=1,
        stem_norm=False,
        base_channels=32,
        num_blocks=(2, 2, 2, 2, 2),
        plane_ratios=(2, 4, 8, 8, 8),
        strides=(1, 2, 2, 2, 2),
        hook_positions=dict(l=(0, 5), g=(0, 1, 2, 3))
    ),
    decoder=dict(
        type='GuidedDecoder',
        num_stage=5,
        base_channels=32,
        plane_ratios=(8, 8, 8, 4, 2, 1),
        strides=(2, 2, 2, 2, 1),
        hook_positions=dict(l=(4,), g=(0, 1, 2, 3))
    ),
    refinement=dict(
        type='LRRURefinement',
    ),
    loss_cfg=dict(type='ComposeLoss', 
            loss_cfgs=[dict(type='CascadeNormLoss', p=1, min_depth=0.0, max_depth=90.0, normalize_type='mean'),
                        dict(type='CascadeNormLoss', p=2, min_depth=0.0, max_depth=90.0, normalize_type='mean')],
            loss_weights=[1.0, 1.0],
            loss_names=['l1loss', 'l2loss']
    )
)