# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        T_max=200,  # maximal epochs
        eta_min=1e-4,  # minimal lr
        by_epoch=True)
]
# training schedule for 100 epochs
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=200)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    logger=dict(type='LoggerHook', log_metric_by_epoch=True, interval=1e9, ignore_last=False, interval_exp_name=0),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=1e5, save_best='mIoU', rule='greater'),
    visualization=dict(type='SegVisualizationHook'))
