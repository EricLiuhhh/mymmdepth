from mmdepth.registry import DATASETS
from mmdepth.models.data_preprocessors import DepthDataPreprocessor
from mmengine.registry import DefaultScope
from mmengine.runner import Runner
dataset_type = 'KittiCompletion'
data_root = 'data/kitti_depth'
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
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='kitti_completion_infos_train.pkl',
        data_prefix={},
        pipeline=train_pipeline,
        test_mode=False)
)
DefaultScope.get_instance('test', scope_name='mmdepth')
# dataset = DATASETS.build(train_dataloader['dataset'])
# dataset.__getitem__(0)
dataloader = Runner.build_dataloader(train_dataloader)
preprocessor = DepthDataPreprocessor()
preprocessor._device = 'cuda:0'
for batch in dataloader:
    preprocessor(batch)