reduce_zero_label = True
DMS_train = dict(type='DMSMTLDataset',
                 which_task='1',
                 data_root='data/DMS',
                 reduce_zero_label=reduce_zero_label,
                 data_prefix=dict(img_path='images/training', seg_map_path='annotations/training'),
                 pipeline=[dict(type='LoadImageFromFile'),
                           dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label),
                           dict(type='RandomResize',
                                scale=(2048, 512),
                                ratio_range=(0.5, 2.0),
                                keep_ratio=True),
                           dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
                           dict(type='RandomFlip', prob=0.5),
                           dict(type='PhotoMetricDistortion'),
                           dict(type='MTLPackSegInputs')])
DMS_val = dict(type='DMSMTLDataset',
               which_task='1',
               data_root='data/DMS',
               reduce_zero_label=reduce_zero_label,
               data_prefix=dict(img_path='images/validation', seg_map_path='annotations/validation'),
               pipeline=[dict(type='LoadImageFromFile'),
                         dict(type='Resize', scale=(2048, 512), keep_ratio=True),
                         dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label),
                         dict(type='MTLPackSegInputs')])
DMS_test = dict(type='DMSMTLDataset',
                which_task='1',
                data_root='data/DMS',
                reduce_zero_label=reduce_zero_label,
                data_prefix=dict(img_path='images/test', seg_map_path='annotations/test'),
                pipeline=[dict(type='LoadImageFromFile'),
                          dict(type='Resize', scale=(2048, 512), keep_ratio=True),
                          dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label),
                          dict(type='MTLPackSegInputs')])

COCO_train = dict(type='COCOStuff10KDataset',
                  which_task='2',
                  data_root='data/coco_stuff10k',
                  reduce_zero_label=reduce_zero_label,
                  data_prefix=dict(img_path='images/train2014', seg_map_path='annotations/train2014'),
                  pipeline=[dict(type='LoadImageFromFile'),
                            dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label),
                            dict(type='RandomResize',
                                 scale=(2048, 512),
                                 ratio_range=(0.5, 2.0),
                                 keep_ratio=True),
                            dict(type='RandomCrop', crop_size=(512, 512), cat_max_ratio=0.75),
                            dict(type='RandomFlip', prob=0.5),
                            dict(type='PhotoMetricDistortion'),
                            dict(type='MTLPackSegInputs')])
COCO_val = dict(type='COCOStuff10KDataset',
                which_task='2',
                data_root='data/coco_stuff10k',
                reduce_zero_label=reduce_zero_label,
                data_prefix=dict(img_path='images/test2014', seg_map_path='annotations/test2014'),
                pipeline=[dict(type='LoadImageFromFile'),
                          dict(type='Resize', scale=(2048, 512), keep_ratio=True),
                          dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label),
                          dict(type='MTLPackSegInputs')])
COCO_test = dict(type='COCOStuff10KDataset',
                 which_task='2',
                 reduce_zero_label=reduce_zero_label,
                 data_root='data/coco_stuff10k',
                 data_prefix=dict(img_path='images/test2014', seg_map_path='annotations/test2014'),
                 pipeline=[dict(type='LoadImageFromFile'),
                           dict(type='Resize', scale=(2048, 512), keep_ratio=True),
                           dict(type='LoadAnnotations', reduce_zero_label=reduce_zero_label),
                           dict(type='MTLPackSegInputs')])

COCO_repeat_train = dict(type='RepeatDataset',
                         times=2,
                         dataset=COCO_train)

DMS_COCO_train = dict(type='ConcatDataset',
                      datasets=[DMS_train, COCO_train])
DMS_COCO_val = dict(type='ConcatDataset',
                    datasets=[DMS_val, COCO_val])
DMS_COCO_test = dict(type='ConcatDataset',
                     datasets=[DMS_test, COCO_test])

train_dataloader = dict(dataset=DMS_COCO_train,
                        batch_size=4,
                        num_workers=4,
                        persistent_workers=True,
                        sampler=dict(type='DefaultSampler', shuffle=True),)
val_dataloader = dict(dataset=DMS_COCO_val,
                      batch_size=1,
                      num_workers=4,
                      persistent_workers=True,
                      sampler=dict(type='DefaultSampler', shuffle=False))
test_dataloader = dict(dataset=DMS_COCO_test,
                       batch_size=1,
                       num_workers=4,
                       persistent_workers=True,
                       sampler=dict(type='DefaultSampler', shuffle=False))

val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='MTLPackSegInputs')]
