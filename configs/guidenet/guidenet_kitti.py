_base_ = [
    '../_base_/datasets/kitti_completion.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
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
    dict(type='RandomCrop', crop_size=(1216, 256)),
    dict(type='PackDepthInputs')
] 
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepth'),
    dict(type='BottomCrop', crop_size=(1216, 256)),
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
    type='GuideNet',
    data_preprocessor=data_preprocessor,
    img_encoder=dict(
        type='GuidedResNet',
        in_channels=3,
        base_channels=32,
        num_blocks=(2, 2, 2, 2, 2),
        plane_ratios=(2, 4, 8, 8, 8),
        strides=(2, 2, 2, 2, 2),
        hook_positions=dict(l=(1, 2, 3, 4, 5))
    ),
    img_decoder=dict(
        type='GuidedDecoder',
        num_stage=4,
        base_channels=32,
        plane_ratios=(8, 8, 8, 4, 2),
        strides=(2, 2, 2, 2),
        hook_positions=dict(g=(0, 1, 2, 3))
    ),
    img_decode_guide=dict(
        guide_cfg=dict(type='AddGuide'),
        guide_map=tuple(zip(('l4', 'l3', 'l2', 'l1'), ('l0', 'l1', 'l2', 'l3')))
    ),
    depth_guide=dict(
        guide_cfg=dict(type='KernelLearningGuide'),
        guide_map=tuple(zip(('g3', 'g2', 'g1', 'g0'), ('l1', 'l2', 'l3', 'l4'))),
    ),
    decode_guide=dict(
        guide_cfg=dict(type='AddGuide'),
        guide_map=tuple(zip(('g3', 'g2', 'g1', 'g0'), ('l0', 'l1', 'l2', 'l3')))
    ),
    depth_encoder=dict(
        type='GuidedResNet',
        in_channels=1,
        stem_norm=False,
        base_channels=32,
        num_blocks=(2, 2, 2, 2, 2),
        plane_ratios=(2, 4, 8, 8, 8),
        strides=(2, 2, 2, 2, 2),
        hook_positions=dict(l=(0, 5), g=(0, 1, 2, 3))
    ),
    decoder=dict(
        type='GuidedDecoder',
        num_stage=5,
        base_channels=32,
        plane_ratios=(8, 8, 8, 4, 2, 1),
        strides=(2, 2, 2, 2, 2),
        hook_positions=dict(l=(4,))
    )
)