import os
import json
import torchvision
import torch
import pkbar
import torchvision.transforms as T
import argparse


def prepare_DMS46_Benchmark(dataset_dir):
    # check legality
    if not dataset_dir.startswith('/'):
        raise ValueError('dataset_dir must be an absolute path!')
    if dataset_dir.endswith('/'):
        dataset_dir = dataset_dir[:-1]

    # get all label files
    label_dir = f'{dataset_dir}/labels/'
    label_files = []
    for each_dir in os.listdir(label_dir):
        for each_file in os.listdir(f'{label_dir}/{each_dir}'):
            label_files.append(f'{label_dir}/{each_dir}/{each_file}')

    # prepare taxonomy file
    bad_categories_idx = [14, 22, 25, 28, 31, 40, 42, 45, 54, 55]
    good_categories_idx_before_mapping = [i for i in range(57) if i not in bad_categories_idx]
    mapping = {15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 23: 21, 24: 22, 26: 23, 27: 24, 29: 25,
               30: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31, 37: 32, 38: 33, 39: 34, 41: 35, 43: 36, 44: 37,
               46: 38, 47: 39, 48: 40, 49: 41, 50: 42, 51: 43, 52: 44, 53: 45, 56: 46}
    taxonomy_file = f'{dataset_dir}/taxonomy.json'
    with open(taxonomy_file, 'r') as f:
        taxonomy = json.load(f)
    categories = taxonomy['shortnames']
    cmap = taxonomy['srgb_colormap']
    good_categories = []
    good_cmap = []
    for cat_idx in good_categories_idx_before_mapping:
        good_categories.append(categories[cat_idx])
        good_cmap.append(cmap[cat_idx])
    good_categories_idx = list(range(47))
    taxonomy_dict = {'semantic_labels': good_categories_idx, 'names': good_categories, 'cmap': good_cmap}
    os.system(f'rm -rf {taxonomy_file}')
    json_object = json.dumps(taxonomy_dict, indent=4)
    with open(f'{taxonomy_file}', 'w') as f:
        f.write(json_object)

    # prepare benchmark labels
    pbar = pkbar.Pbar(name='Preparing benchmark labels, please wait...', target=len(label_files))
    transform = T.ToPILImage()
    for i, each_label in enumerate(label_files):
        label = torchvision.io.read_image(each_label)
        for bad_cat in bad_categories_idx:
            label = torch.where(label == bad_cat, 0, label)
        for mapping_idx in list(mapping.keys()):
            label = torch.where(label == mapping_idx, mapping[mapping_idx], label)
        label = transform(label)
        label.save(each_label)
        pbar.update(i)

    os.system(f'rm -rf {dataset_dir}/info.json')
    os.system(f'rm -rf {dataset_dir}/info.json.gz')
    os.system(f'rm -rf {dataset_dir}/LICENSE')
    os.system(f'rm -rf {dataset_dir}/original_images')
    os.system(f'mv {dataset_dir}/images/train {dataset_dir}/images/training')
    os.system(f'mv {dataset_dir}/labels/train {dataset_dir}/labels/training')
    os.system(f'mv {dataset_dir}/labels {dataset_dir}/annotations')
    os.system(f'mv {dataset_dir} {os.path.abspath(os.path.join(dataset_dir, os.pardir))}/DMS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_dir', help='Directory of the dataset. Probably /your/home/dir/DMS_v1/')
    args = parser.parse_args()
    prepare_DMS46_Benchmark(args.dataset_dir)
