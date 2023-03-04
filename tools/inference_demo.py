from materobot.apis import inference_moe_model, inference_single_model, init_model, show_result_pyplot
from mmseg.utils import register_all_modules
import os

img_path = 'data/coco_stuff10k/images/train2014/COCO_train2014_000000012107.jpg'
# config_path = 'materobot/configs/matevit_vit-t_single-task_dms.py'
# checkpoint_path = 'work_dirs/matevit_vit-t_single-task_dms/best_mIoU_epoch_89.pth'
# config_path = 'materobot/configs/matevit_vit-t_single-task_coco.py'
# checkpoint_path = 'work_dirs/matevit_vit-t_single-task_coco/best_mIoU_epoch_98.pth'
config_path = 'materobot/configs/matevit_vit-t_multi-task.py'
checkpoint_path = 'work_dirs/matevit_vit-t_multi-task/best_mIoU_epoch_196.pth'

# register all modules in mmseg into the registries
register_all_modules()

# build the model from a config file and a checkpoint file
model = init_model(config_path, checkpoint_path, device='cuda:0')

# inference on given image
# result = inference_single_model(model, img_path)
result = inference_moe_model(model, img_path)

# print('Prediction:')  # print single dms/coco model data
# print(result[0].pred_sem_seg.data)  # result[0].pred_sem_seg.data.shape == (bs=1, h, w)
print('Task 1 (DMS) prediction:')  # print moe model data
print(result[0].pred_sem_seg.data)  # result[0].pred_sem_seg.data.shape == (bs=1, h, w)
print('Task 2 (COCO) prediction:')  # print moe model data
print(result[1].pred_sem_seg.data)  # result[1].pred_sem_seg.data.shape == (bs=1, h, w)

# visualize results
os.system(f'cp {img_path} work_dirs/ori.png')
# show_result_pyplot(model, img_path, result[0], show=False, opacity=0.8, out_file='work_dirs/dms.png', save_dir='work_dirs')  # visualize single dms model
# show_result_pyplot(model, img_path, result[0], show=False, opacity=0.8, out_file='work_dirs/coco.png', save_dir='work_dirs')  # visualize single coco model
show_result_pyplot(model, img_path, result[0], show=False, opacity=0.8, out_file='work_dirs/dms.png', save_dir='work_dirs')  # visualize moe model
show_result_pyplot(model, img_path, result[1], show=False, opacity=0.8, out_file='work_dirs/coco.png', save_dir='work_dirs') # visualize moe model
