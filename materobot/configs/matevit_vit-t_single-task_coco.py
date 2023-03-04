_base_ = [
    './_base_/models/matevit_single-task.py',
    './_base_/datasets/coco-stuff10k.py', './_base_/default_runtime.py',
    './_base_/schedules/schedule_100epochs.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_tiny_p16_384_20220308-cce8c795.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(embed_dims=192,
                  num_heads=3,
                  ds_keep_ratio=0.95),
    decode_head=dict(
        type='SegmenterMaskTransformerHead',
        num_classes=171,
        in_channels=192,
        channels=192,
        num_heads=3,
        embed_dims=192))

optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=2)
val_dataloader = dict(batch_size=1)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)
