import os
import gzip
import shutil
import json
from urllib import request
import pkbar
import argparse

"""
After downloading DMS dataset, you need to follow
this instruction(https://github.com/apple/ml-dms-dataset#preparing-the-color-images) to prepare RGB images
"""


def download_DMS(save_dir):

    url = 'https://docs-assets.developer.apple.com/ml-research/datasets/dms/dms_v1_labels.zip'
    dms_root = os.path.join(save_dir, 'DMS_v1')
    zip_file = os.path.join(os.getcwd(), 'dms_v1_labels.zip')
    info_zip_file = os.path.join(dms_root, "info.json.gz")
    info_file = os.path.join(dms_root, 'info.json')
    ori_imgs_dir = os.path.join(dms_root, 'original_images')

    # check legality
    if not save_dir.startswith('/'):
        raise ValueError('Please use absolute dir!')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # download and unzip
    print('Downloading labels, please wait...')
    os.system(f'wget {url}')
    os.system(f'unzip {zip_file} -d {save_dir}')
    os.system(f'rm -rf {zip_file}')
    with gzip.open(info_zip_file, 'rb') as f_in:
        with open(info_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # download RGB
    with open(info_file, 'r') as f:
        info = json.load(f)
    ori_imgs_list = []
    for each in info:
        ori_imgs_list.append(each['openimages_metadata']['OriginalURL'])
    os.makedirs(ori_imgs_dir, exist_ok=True)
    pbar = pkbar.Pbar(name='Downloading images, please wait...', target=len(ori_imgs_list))
    for i, each_url in enumerate(ori_imgs_list):
        download_image(each_url, ori_imgs_dir)
        pbar.update(i)


def download_image(image_url, image_dir):
    try:
        request.urlretrieve(image_url, f'{image_dir}/{image_url.split("/")[-1]}')
    except:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('save_dir', help='Directory to save DMS dataset. Should be an absolute path.')
    args = parser.parse_args()
    download_DMS(args.save_dir)
