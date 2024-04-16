# training schedule for 1x
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=2)
val_cfg = dict(type='ValLoop')
# val_cfg = None
# test_cfg = dict(type='TestLoop')
test_cfg = None
# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=True, begin=0, end=1, convert_to_iter_based=True),
    dict(
        type='MultiStepLR',
        begin=1,
        end=20,
        by_epoch=True,
        milestones=[10, 20],
        gamma=0.2)
]

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='Adam', lr=0.001, weight_decay=0.0))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
