# --*-- UTF-8- --*--
# Data  :  下午9:00
# Name  : data_prepare_hongkong.py

import os.path

import laspy
import yaml
import pickle
import argparse
import numpy as np
from pathlib import Path
from sklearn.neighbors import KDTree
import math
import random
from helper_tool import DataProcessing as DP
from helper_ply import write_ply, read_ply


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path',  default='DATASET_PATH',
                        help='source dataset path [default: None]')
    parser.add_argument('--dst_path',  default='DATASET_PATH',
                        help='destination dataset path [default: None]')
    parser.add_argument('--grid_size', type=float, default=-1, help='Subsample Grid Size, -1 means no gird sumple[default: -1]')
    parser.add_argument('--yaml_config', default='hongkong_config.yaml', help='data config yaml file path')
    parser.add_argument('--shuffle', type=bool, default=True, help='if random split datasets[default: True]')
    parser.add_argument('--remap_label', type=bool, default=False, help='if remap labels [default: False]')
    args = parser.parse_args()

    # split = ['B_1']
    # split = 'Track_C'
    split = None

    # mkdir save file path
    ori_path = Path(args.src_path)
    dst_path = Path(args.dst_path) if args.dst_path is not None else ori_path.parent / ('subsampled_{.3f}'.format(args.grid_size))
    dst_path.mkdir(parents=True, exist_ok=True)
    data_path = dst_path / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
    kdtree_path = dst_path / 'kdtree'
    kdtree_path.mkdir(parents=True, exist_ok=True)
    proj_path = dst_path / 'proj'
    proj_path.mkdir(parents=True, exist_ok=True)


    # load data config
    with open(args.yaml_config, 'r') as f:
        DATA = yaml.safe_load(f)

    # get train-val-test file list and save into a txt and a yaml file is better.
    split_dic = dict()
    train_list = []
    val_list = []
    test_list = []

    all_list = list(ori_path.glob('*.ply'))

    if split is not None:
        for i, file in enumerate(all_list):
            filename = file.stem
            if split in filename:
                val_list.append(file)
                test_list.append(file)
            else:
                train_list.append(file)
    else:
        split = DATA['split_all']

        if args.shuffle:
            random.shuffle(all_list)
        train_idx = math.floor(len(all_list) * split[0][0])
        val_idx = math.floor(len(all_list) * (split[0][0] + split[0][1]))

        train_list = all_list[: train_idx]
        val_list = all_list[train_idx: val_idx]
        test_list = all_list[val_idx:]
    print('train: {}, val: {}, test: {}'.format(len(train_list), len(val_list), len(test_list)))

    # save train/val list -> save name
    split_dic['train'] = [str(i.stem) for i in train_list]
    split_dic['val'] = [str(i.stem) for i in val_list]
    split_dic['test'] = [str(i.stem) for i in test_list]
    with open(os.path.join(args.dst_path, 'data_split_HongKong.yaml'), 'w') as f:
        yaml.dump(split_dic, f)


    for i, pc_path in enumerate(all_list):
        pc_name = pc_path.stem

        sub_ply_file = data_path / (pc_name + '.ply')
        sub_kdtree_file = kdtree_path / (pc_name + '_KDTree.pkl')
        proj_file = proj_path / (pc_name + '_proj.pkl')

        if sub_ply_file.exists() and sub_kdtree_file.exists() and proj_file.exists():
            continue

        # laod data
        # points, label, intensity = DP.read_las_file(pc_path)
        if pc_path.suffix == '.txt':
            data = np.loadtxt(str(pc_path))
            point = data[:, 0:3].astype(np.float32)
            colors = data[:, 3:6].astype(np.uint8)
            label = data[:, -1].astype(np.int32)
            intensity = data[:, 6].astype(np.float32).reshape(-1, 1)
        elif pc_path.suffix == '.ply':
            data = read_ply(str(pc_path))
            point = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
            colors = np.vstack((data['red'], data['green'], data['blue'])).T.astype(np.uint8)
            intensity = data['intensity'].reshape(-1, 1)
            label = data['class'].astype(np.int32)
        elif pc_path.suffix == '.las' or pc_path.suffix == '.laz':
            las = laspy.read(str(pc_path))
            point = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
            colors = np.vstack((las.red, las.green, las.blue)).T
            label = np.array(las.classification).astype(np.int32)
            intensity = las.intensity.reshape(-1, 1)

            # covert uint16 to uint8
            colors = np.ceil(colors / 65536 * 256)
            colors = colors.astype(np.uint8)
        else:
            raise NotImplementedError

        # subsample
        if args.grid_size > 0:
            sub_xyz, sub_colors, sub_labels = DP.grid_sub_sampling(point, features=colors, labels=label, grid_size=args.grid_size)
            _, sub_intensity = DP.grid_sub_sampling(point, features=intensity, grid_size=args.grid_size)

            sub_colors = sub_colors.astype(np.uint8)
            sub_labels = sub_labels.astype(np.int32)
            sub_intensity = sub_intensity.astype(np.float32)
            write_ply(str(sub_ply_file), [sub_xyz, sub_colors, sub_intensity, sub_labels],
                      ['x', 'y', 'z', 'red', 'green', 'blue', 'intensity', 'class'])
            print("Successful save subsampled data in %s" % (sub_ply_file))

        search_tree = KDTree(point, leaf_size=50)

        with open(sub_kdtree_file, 'wb') as f:
            pickle.dump(search_tree, f)
        print("Successful save kdtree file in %s" % (sub_kdtree_file))

        proj_idx = np.squeeze(search_tree.query(point, return_distance=False))
        proj_idx = proj_idx.astype(np.int32)
        with open(proj_file, 'wb') as f:
            pickle.dump([proj_idx, label], f)
        print("Successful save proj file in %s" % (proj_file))
