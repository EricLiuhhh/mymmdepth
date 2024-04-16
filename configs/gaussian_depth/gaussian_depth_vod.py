_base_ = [
    '../_base_/datasets/vod_completion.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
# import mmdet3d.datasets.transforms
# 使用 custom_imports 将 mmdet3d 的 transforms 添加进注册器
custom_imports = dict(imports=['mmdet3d.datasets.transforms'], allow_failed_imports=False)
point_type = {{_base_.point_type}}
point_dim = {{_base_.point_dim}}
backend_args = None
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='KeyMapper',
         mapping = {
            'lidar_points': 'input_points'
         },
         remapping = {
             'lidar_points': 'input_points',
             'points': 'points'
         },
         transforms=[
             dict(type='mmdet3d.LoadPointsFromFile', coord_type='LIDAR', load_dim=point_dim, use_dim=point_dim),
         ]),
    dict(type='LoadDepth'),
    dict(type='LoadCalibVoD'),
    dict(type='PackDepthInputs')
] 
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepth'),
    dict(type='LoadCalibVoD'),
    dict(type='PackDepthInputs')
] 
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))

data_preprocessor = dict(
    type='DepthDataPreprocessor',
    voxel=True,
    voxel_type='minkunet',
    batch_first=True,
    max_voxels=80000,
    voxel_layer=dict(
        max_num_points=-1,
        point_cloud_range=[0, -25.6, -3, 51.2, 25.6, 2],
        voxel_size=[0.05, 0.05, 0.05],
        max_voxels=(-1, -1)),
    mean=(0, 0, 0),
    std=(255, 255, 255),
    bgr_to_rgb=True)

model = dict(
    type='GaussianDepth',
    point_encoder=dict(
        type='MinkUNetBackbone',
        in_channels=point_dim,
        num_stages=4,
        base_channels=32,
        encoder_channels=[32, 64, 128, 256],
        encoder_blocks=[2, 2, 2, 2],
        decoder_channels=[256, 128, 96, 96],
        decoder_blocks=[2, 2, 2, 2],
        block_type='basic',
        sparseconv_backend='spconv'      
        ),
    prefiller=dict(
        type='GaussianPrefiller'
    )
)