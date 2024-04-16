_base_ = [
    '../_base_/datasets/kitti_completion.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
backend_args = None
train_pipeline = [
    dict(type='LoadTripletImages', backend_args=backend_args),
    dict(type='LoadDepth'),
    dict(type='LoadCalibKitti'),
    dict(type='Normalize', mean=[0, 0, 0], std=[255, 255, 255], to_rgb=False),
    dict(type='PackDepthInputs')
] 
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadDepth'),
    dict(type='LoadCalibKitti'),
    dict(type='Normalize', mean=[0, 0, 0], std=[255, 255, 255], to_rgb=False),
    dict(type='PackDepthInputs')
] 
train_dataloader = dict(dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(dataset=dict(pipeline=val_pipeline))

data_preprocessor = dict(
    type='KBNetDataPreprocessor',
    bgr_to_rgb=True)

model = dict(
    type='KBNet',
    data_preprocessor=data_preprocessor,
    prefiller=dict(
        type='SparseToDensePool',
        input_channels=2,
        n_filter=8,
        bias=False
    ),
    img_branch=dict(
        type='KBNetEncoder',
        in_channels=3,
    ),
    depth_branch=dict(
        type='KBNetEncoder',
        in_channels=8,
        ext_feats_channels=3,
        coord_guide=True,
        planes=[16, 32, 64, 128, 128],
    ),
    backproj_layer=dict(
        type='CalibratedBackprojectionBlocks'
    ),
    decoder=dict(
        type='KBNetDecoder'
    )
)