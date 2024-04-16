_base_ = [
    '../_base_/datasets/kitti_completion.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepth'),
    dict(type='LoadCalibKitti'),
    dict(type='GenPosEmbd', height=352, width=1216, scale=2, offset=-1),
    dict(type='ColorJitter', 
        brightness=0.4,
        contrast=0.4,
        saturation=0.4),
    dict(type='BottomCrop', crop_size=(1216, 352)),
    dict(type='RandomFlipBundle', prob=0.5),
    dict(type='RandomCrop', crop_size=(1216, 256)),
    dict(type='PackDepthInputs')
] 
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepth'),
    dict(type='LoadCalibKitti'),
    dict(type='GenPosEmbd', height=352, width=1216, scale=2, offset=-1),
    dict(type='PackDepthInputs')
] 
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))

data_preprocessor = dict(
    type='PENetDataPreprocessor',
    bgr_to_rgb=True,
    img_shape=(352, 1216),
    num_stage=6)

model = dict(
    type='PENet',
    data_preprocessor=data_preprocessor,
    img_branch=dict(
        type='ENetEncoder',
        base_channels=32,
        conv_out_cfg=dict(type='ConvT', output_padding=0)
    ),
    depth_guide=dict(
        guide_cfg=[dict(type='CatGuide',
        normal_order=False)] * 4 + [dict(type='AddGuide')],
        guide_map=tuple(zip(('l9', 'l8', 'l7', 'l6', 'l5'), ('l1', 'l2', 'l3', 'l4', 'l5')))
    ),
    depth_branch=dict(
        type='ENetEncoder',
        in_channels=2,
        base_channels=32,
        plane_ratios=(2, 1, 1, 1, 1),
        custom_inplanes=(32, 128, 256, 512, 1024),
        self_guide_map=dict([(f'l{i}', f'l{5*2-i}') for i in range(1, 5)]),
        hook_before_self_guide=True
    ),
    refinement=dict(
        type='DACSPNPP'
    )
)