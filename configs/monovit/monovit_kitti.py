_base_ = [
    '../_base_/datasets/kitti_estimation.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
backend_args = None
input_shape = (1280, 384)
batch_size = 2
train_pipeline = [
    dict(type='LoadTripletImages', backend_args=backend_args),
    #dict(type='LoadDepth'),
    dict(type='GenGTDepthKitti', shape=(1242, 375)),
    dict(type='LoadCalibKitti', K=[[0.58*1242, 0, 0.5*1242],
                                    [0, 1.92*375, 0.5*375],
                                    [0, 0, 1]]),
    dict(type='RandomFlipBundle', prob=0.5),
    dict(type='ResizeBundle', size=input_shape, interpolation='lanczos'),
    dict(type='BuildImagePyramid', num_scales=4),
    dict(type='ColorJitter', 
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1),
    dict(type='PackDepthInputs')
] 
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='ResizeBundle', size=input_shape, interpolation='lanczos'),
    #dict(type='LoadDepth'),
    dict(type='GenGTDepthKitti', shape=(1242, 375), keep_velo_depth=True),
    dict(type='PackDepthInputs')
] 
train_dataloader = dict(batch_size=batch_size, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=batch_size, dataset=dict(pipeline=val_pipeline))

data_preprocessor = dict(
    type='DepthDataPreprocessor',
    mean=(0, 0, 0),
    std=(255, 255, 255),
    bgr_to_rgb=True)

model = dict(
    type='HRDepth',
    input_shape=input_shape,
    batch_size=batch_size,
    data_preprocessor=data_preprocessor,
    img_encoder=dict(
        type='MPViT',
        num_stages=4,
        num_path=[2, 3, 3, 3],
        num_layers=[1, 3, 6, 3],
        embed_dims=[64, 128, 216, 288],
        mlp_ratios=[4, 4, 4, 4],
        num_heads=[8, 8, 8, 8],
    ),
    img_decoder=dict(
        type='HRDepthDecoder',
        use_input_attention=True,
        input_channels=[64, 128, 216, 288, 288],
        num_ch_enc=[64, 64, 128, 256, 512],
        padding_mode='reflect',
        norm_cfg=None,
        act_cfg=dict(type='ELU')
    ),
    # pose_encoder=dict(
    #     type='StdResNet',
    #     depth=18,
    #     num_input_images=2,
    #     hook_positions=dict(l=(0, 2, 3, 4, 5))
    # ),
    # pose_decoder=dict(
    #     type='PoseDecoder',
    #     input_channels=512,
    #     squeeze_channels=256,
    #     n_filters=[256, 256],
    #     stride=1,
    #     act_out=False
    # )
)