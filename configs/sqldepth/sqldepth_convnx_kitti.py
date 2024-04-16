_base_ = [
    '../_base_/datasets/kitti_estimation.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
backend_args = None
input_shape = (1024, 320)
batch_size = 8
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
    type='SQLDepth',
    input_shape=input_shape,
    batch_size=batch_size,
    data_preprocessor=data_preprocessor,
    img_encoder=dict(
        type='ConvNeXtL',
    ),
    img_decoder=dict(
        type='GuidedDecoder',
        num_stage=6,
        base_channels=1,
        plane_ratios=((1536, 1536), (1024, 1024), (512, 512), (256, 256), (256, 128), (128, 128)),
        strides=(2, 2, 2, 2, 1, 1),
        upsample_type='conv_bilinear',
        input_upsample_type='bilinear',
        hook_positions=dict(g=(1, 2, 3), l=(5,)),
    ),
    img_decode_guide=dict(
        guide_cfg=[dict(type='CatConvGuide', out_planes=1024),
                   dict(type='CatConvGuide', out_planes=512),
                   dict(type='CatConvGuide', out_planes=256)],
        guide_map=tuple(zip(('l4', 'l3', 'l2'), ('l0', 'l1', 'l2')))
    ),
    conv_out_locations=('l5',),
    conv_out_cfg=dict(ks=1, out_channels=32),
    sql_layer=dict(
        type='SQLLayer',
        in_channels=32,
        patch_size=32,
        dim_out=64,
        embedding_dim=32,
        query_nums=64, 
        num_heads=4,
        dim_feedforward=1024,
        min_val=0.001, 
        max_val=80.0
    )
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