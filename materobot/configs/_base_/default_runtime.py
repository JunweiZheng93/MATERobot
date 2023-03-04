default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=True, custom_cfg=[dict(data_src='loss',
                                                     method_name='mean',
                                                     window_size='epoch'),
                                                dict(data_src='decode.loss_ce',
                                                     method_name='mean',
                                                     window_size='epoch'),
                                                dict(data_src='decode.acc_seg',
                                                     method_name='mean',
                                                     window_size='epoch')])
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')
