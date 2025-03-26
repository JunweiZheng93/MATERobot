# MATERobot: Material Recognition in Wearable Robotics for People with Visual Impairments

<p>
<a href="https://arxiv.org/pdf/2302.14595v1.pdf">
    <img src="https://img.shields.io/badge/PDF-arXiv-brightgreen" /></a>
<a href="https://junweizheng93.github.io/publications/MATERobot/MATERobot.html">
    <img src="https://img.shields.io/badge/Project-Homepage-red" /></a>
<a href="https://pytorch.org/get-started/previous-versions/#linux-and-windows">
    <img src="https://img.shields.io/badge/Framework-PyTorch%201.12.1-orange" /></a>
<a href="https://github.com/open-mmlab/mmsegmentation/tree/1.x">
    <img src="https://img.shields.io/badge/Framework-mmsegmentation%201.x-yellowgreen" /></a>
<a href="https://github.com/JunweiZheng93/MATERobot/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" /></a>
</p>

## 🏡 Project Homepage

This project has been selected as the Best Paper Finalist on Human-Robot Interaction at ICRA 2024! For more information about the project, please refer to our [project homepage](https://junweizheng93.github.io/publications/MATERobot/MATERobot.html).

## 🔧 Setup

```shell
# create virtual environment
conda create -n materobot python=3.8 -y
conda activate materobot
# install PyTorch
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.6 -c pytorch -c conda-forge -y
# install other packages
pip install -U openmim
mim install mmengine==0.4.0
mim install mmcv==2.0.0rc3
pip install -r requirements.txt
```

## 📚 Prepare datasets

The overall folder structure is shown below:

```text
MATERobot
├── mmseg
├── materobot
├── pretrain
├── requirements
├── tools
├── data
│   ├── DMS
│   │   ├── images
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   │   ├── annotations
│   │   │   ├── training
│   │   │   ├── validation
│   │   │   ├── test
│   ├── coco_stuff10k
│   │   ├── images
│   │   │   ├── train2014
│   │   │   ├── test2014
│   │   ├── annotations
│   │   │   ├── train2014
│   │   │   ├── test2014
│   │   ├── imagesLists
│   │   │   ├── train.txt
│   │   │   ├── test.txt
│   │   │   ├── all.txt
```

### DMS

Download [DMS](https://arxiv.org/abs/2207.10614) dataset using `tools/dms_download_tools/download_DMS.py`:

```python
python tools/dms_download_tools/download_DMS.py /home/usr_name/project_root/data
```

After downloading, prepare the DMS dataset according to its [official GitHub repository](https://github.com/apple/ml-dms-dataset#sample-code).
You need to solve all problems by yourself according to the `preparation_outcomes.json` and `image_issues.json`.
Make sure every label has its corresponding image (the number of labels should be equal to the number of images).

Finally, prepare DMS labels in order to get the desired DMS46 dataset:

```python
python tools/dms_download_tools/prepare_DMS_labels.py /home/usr_name/project_root/data/DMS_v1
```

### COCO-Stuff10k

Please prepare coco-stuff10k according to [this page](https://github.com/open-mmlab/mmsegmentation/blob/1.x/docs/en/user_guides/2_dataset_prepare.md#coco-stuff-10k).

## 🖥 Prepare pretrained backbone

Please download the pretrained backbone model [here](https://drive.google.com/drive/folders/1TIF5ZUXWRB7688l8l2-KVBJGZVzBCuaV?usp=share_link)
and place the model to `pretrain` folder under the project root.

## 📦 Usage

### Train

#### Single-task Model

```shell
# Command: bash tools/dist_train.sh config/file/path num_gpus
bash tools/dist_train.sh materobot/configs/matevit_vit-t_single-task_dms.py 4
```

#### Multi-task Model

```shell
# Command: bash tools/dist_train.sh config/file/path num_gpus
bash tools/dist_train.sh materobot/configs/matevit_vit-t_multi-task.py 4
```

### Test

#### Single-task Model

```shell
# Command: bash tools/dist_test.sh config/file/path checkpoint/file/path num_gpus
bash tools/dist_test.sh materobot/configs/matevit_vit-t_single-task_dms.py work_dirs/matevit_vit-t_single-task_dms/best_mIoU_epoch_100.pth 4
```

#### Multi-task Model

Since there is only one kind of gt label for each input data, you need to modify the dataset before running the test script:

```python
# dataset before modification in materobot/configs/matevit_vit-t_multi-task.py:
_base_ = [
    './_base_/models/matevit_multi-task.py',
    './_base_/datasets/dms_coco.py', './_base_/default_runtime.py',
    './_base_/schedules/schedule_200epochs.py'
]

# if you want to test on DMS dataset:
_base_ = [
    './_base_/models/matevit_multi-task.py',
    './_base_/datasets/dms.py', './_base_/default_runtime.py',
    './_base_/schedules/schedule_200epochs.py'
]

# if you want to test on COCO-Stuff10k dataset:
_base_ = [
    './_base_/models/matevit_multi-task.py',
    './_base_/datasets/coco-stuff10k.py', './_base_/default_runtime.py',
    './_base_/schedules/schedule_200epochs.py'
]
```

After the modification, you can run the following command:

```shell
# Command: bash tools/dist_test.sh config/file/path checkpoint/file/path num_gpus
bash tools/dist_test.sh materobot/configs/matevit_vit-t_multi-task.py work_dirs/matevit_vit-t_multi-task/best_mIoU_epoch_200.pth 4
```

### Inference

Please refer to [inference_demo.py](tools/inference_demo.py)

## 🖥 Checkpoints

Download from [here](https://drive.google.com/drive/folders/1yN-lUu5DLcIYmWum6cMlrkWW5b2juPZB?usp=sharing)

## 📖 Citation

If you are interested in this work, please cite as below:

```text
@inproceedings{zheng2024materobot,
title={MATERobot: Material Recognition in Wearable Robotics for People with Visual Impairments},
author={Zheng, Junwei and Zhang, Jiaming and Yang, Kailun and Peng, Kunyu and Stiefelhagen, Rainer},
booktitle={2024 IEEE International Conference on Robotics and Automation (ICRA)},
year={2024}
}
```
