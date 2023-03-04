_base_ = [
    './_base_/models/matevit_multi-task.py',
    './_base_/datasets/dms_coco.py', './_base_/default_runtime.py',
    './_base_/schedules/schedule_200epochs.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segmenter/vit_tiny_p16_384_20220308-cce8c795.pth'  # noqa

model = dict(
    type='MTLEncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=checkpoint,
    backbone=dict(embed_dims=192,
                  num_heads=3,
                  ds_keep_ratio=0.95),
    neck=dict(type='MoE',
              input_size=192,
              output_size=192,
              hidden_size=(256, 512),
              num_experts=10,
              k=5,
              loss_coef=1),
    decode_head1=dict(
        type='SegmenterMaskTransformerHead',
        num_classes=46,
        in_channels=192,
        channels=192,
        num_heads=3,
        embed_dims=192),
    decode_head2=dict(
        type='SegmenterMaskTransformerHead',
        num_classes=171,
        in_channels=192,
        channels=192,
        num_heads=3,
        embed_dims=192)
)

find_unused_parameters = True
optimizer = dict(lr=0.001, weight_decay=0.0)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer)
train_dataloader = dict(batch_size=1)
val_dataloader = dict(batch_size=1)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(vis_backends=vis_backends)
