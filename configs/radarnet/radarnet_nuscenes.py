_base_ = [
    '../_base_/datasets/nuscenes_completion_rd.py', '../_base_/default_runtime.py', '../_base_/schedules/schedule_1x.py'
]
backend_args = None
input_shape = (800, 450)
data_preprocessor = dict(
    type='DepthDataPreprocessor',
    mean=(0, 0, 0),
    std=(255, 255, 255),
    bgr_to_rgb=True)

model = dict(
    type='RadarNet',
)