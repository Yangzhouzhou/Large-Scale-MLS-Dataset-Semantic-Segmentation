# --*-- UTF-8- --*--
# Name  : hongkong_dataset.py
# Data  : 2023/4/14 下午4:41
# Description: HongKong dataset

import yaml

from helper_tool import DataProcessing as DP
from helper_tool import ConfigHongKong as cfg
import numpy as np
import time, pickle
from os.path import join
from utils.helper_ply import read_ply
from torch.utils.data import DataLoader, Dataset
import torch
from pathlib import Path
from sklearn.neighbors import KDTree

# read the subsampled data and divide the data into training and validation
class HongKong(Dataset):
    def __init__(self, val_split=None, path=None):
        self.name = 'HongKong'
        if path is None:
            self.path = 'DATASET_PATH'
        else:
            self.path = path
        self.config_path = 'utils/hongkong_config.yaml'
        self.ignored_labels = np.array([])  # ignored labels

        self.init_label()

        # load data split file
        with open(join(self.path, 'data_split_HongKong.yaml'), 'r') as f:
            self.file_names = yaml.safe_load(f)

        if val_split is not None:
            self.train_names, self.val_split = self.get_file_names(val_split)
        else:
            self.val_split = self.file_names['val']
            self.train_names = self.file_names['train']
        self.all_names = self.train_names + self.val_split
        self.size = len(self.all_names)

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_features = {'training': [], 'validation': []}    # R, G, B, intensity
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}

        self.load_sub_sampled_clouds(cfg.sub_grid_size)

        print('Size of training : ', len(self.input_features['training']))  
        print('Size of validation : ', len(self.input_features['validation'])) 

    def load_sub_sampled_clouds(self, sub_grid_size):

        for i, cloud_name in enumerate(self.all_names):
            t0 = time.time()
            if cloud_name in self.val_split:  
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(self.path, 'kdtree', '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(self.path, 'data', '{:s}.ply'.format(cloud_name))

            # read ply data
            data = read_ply(sub_ply_file)
            sub_points = np.vstack((data['x'], data['y'], data['z'])).T.astype(np.float32)
            sub_labels = data['class']
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_intensity = data['intensity'].reshape(-1, 1) / 255
            sub_features = np.concatenate((sub_colors, sub_intensity), axis=1).astype(np.uint8)
            sub_labels = np.array([self.label_to_idx[l] for l in sub_labels]).astype(np.int32)

            # check kdtree file exists
            if not Path(kd_tree_file).exists():
                print('KDTree file not found for cloud {:s}, rebuilding'.format(cloud_name))
                # build kdtree
                search_tree = KDTree(sub_points, leaf_size=50)
                print('KDTree build done in {:.1f}s'.format(time.time() - t0))
                # save kdtree file
                with open(kd_tree_file, 'wb') as f:
                    pickle.dump(search_tree, f)
                print('KDTree saved in {:.1f}s'.format(time.time() - t0))
            else:
                # Read pkl with search tree
                with open(kd_tree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]  
            self.input_features[cloud_split] += [sub_features]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices       
        for i, cloud_name in enumerate(self.val_split):
            t0 = time.time()

            # Validation projection and labels
            proj_file = join(self.path, 'proj', '{:s}_proj.pkl'.format(cloud_name))
            if not Path(proj_file).exists():
                print("This file don't exist: ", proj_file)
                continue

            with open(proj_file, 'rb') as f:
                proj_idx, labels = pickle.load(f)
            self.val_proj += [proj_idx]  
            self.val_labels += [labels]
            print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    def __getitem__(self, idx):
        pass

    def __len__(self):
        # Number of clouds
        return self.size

    # 初始化标签
    def init_label(self):
        with open(self.config_path, 'r') as stream:
            self.DATA = yaml.safe_load(stream)

        self.label_to_names = self.DATA['labels']
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])  
        self.label_to_idx = {l: i for i, l in enumerate(
            self.label_values)}  # {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12}

        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        cfg.class_weights = DP.get_class_weights('HongKong')
        cfg.name = 'HongKong'

    # get file list
    def get_file_names(self, split):
        train_names = []
        val_names = []
        all_data_file = list((Path(self.path + '/data').glob('*.ply')))
        for i, files in enumerate(all_data_file):
            name = files.stem
            if split in name:
                val_names.append(name)
            else:
                train_names.append(name)
        return train_names, val_names


class HongKongSampler(Dataset):

    def __init__(self, dataset, split='training'):
        self.dataset = dataset
        self.split = split
        self.possibility = {}
        self.min_possibility = {}

        if split == 'training':
            self.num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            self.num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        for i, tree in enumerate(self.dataset.input_trees[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]  
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]  
    

    def __getitem__(self, item):
        selected_pc, selected_labels, selected_idx, cloud_ind = self.spatially_regular_gen(item, self.split)
        return selected_pc, selected_labels, selected_idx, cloud_ind

    def __len__(self):

        return self.num_per_epoch
        # return 2 * cfg.val_batch_size

    def spatially_regular_gen(self, item, split):

        # Choose a random cloud         
        cloud_idx = int(np.argmin(self.min_possibility[split]))

        # choose the point with the minimum of possibility in the cloud as query point  
        point_ind = np.argmin(self.possibility[split][cloud_idx])

        # Get all points within the cloud from tree structure   
        points = np.array(self.dataset.input_trees[split][cloud_idx].data, copy=False)

        # Center point of input region  
        center_point = points[point_ind, :].reshape(1, -1)

        # Add noise to the center point
        noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
        pick_point = center_point + noise.astype(center_point.dtype)  
        # Check if the number of points in the selected cloud is less than the predefined num_points
        if len(points) < cfg.num_points:  
            # Query all points within the cloud
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
        else:
            # Query the predefined number of points
            queried_idx = self.dataset.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

        # Shuffle index
        queried_idx = DP.shuffle_idx(queried_idx)  
        # Get corresponding points and colors based on the index
        queried_pc_xyz = points[queried_idx]  
        queried_pc_xyz = queried_pc_xyz - pick_point  
        queried_pc_features = self.dataset.input_features[split][cloud_idx][queried_idx]
        queried_pc_labels = self.dataset.input_labels[split][cloud_idx][queried_idx]

        # Update the possibility of the selected points
        dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)  
        delta = np.square(1 - dists / np.max(dists))  
        self.possibility[split][cloud_idx][queried_idx] += delta 
        self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx])) 

        # up_sampled with replacement
        if len(points) < cfg.num_points: 
            queried_pc_xyz, queried_pc_intensity, queried_idx, queried_pc_labels = \
                DP.data_aug(queried_pc_xyz, queried_pc_features, queried_pc_labels, queried_idx, cfg.num_points)

        queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()  
        queried_pc_features = torch.from_numpy(queried_pc_features).float()
        queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
        queried_idx = torch.from_numpy(queried_idx).float()
        cloud_idx = torch.from_numpy(np.array([cloud_idx], dtype=np.int32)).float()

        points = torch.cat((queried_pc_xyz, queried_pc_features), 1)

        return points, queried_pc_labels, queried_idx, cloud_idx

    def tf_map(self, batch_xyz, batch_features, batch_label, batch_pc_idx, batch_cloud_idx):  
        batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(cfg.num_layers): 
            neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n)  
            sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]  
            pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]  
            up_i = DP.knn_search(sub_points, batch_xyz, 1)  
            input_points.append(batch_xyz)
            input_neighbors.append(neighbour_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            batch_xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [batch_features, batch_label, batch_pc_idx, batch_cloud_idx]

        return input_list


    def collate_fn(self, batch):

        selected_pc, selected_labels, selected_idx, cloud_ind = [], [], [], []
        for i in range(len(batch)):
            selected_pc.append(batch[i][0])
            selected_labels.append(batch[i][1])
            selected_idx.append(batch[i][2])
            cloud_ind.append(batch[i][3])

        selected_pc = np.stack(selected_pc)  
        selected_labels = np.stack(selected_labels)
        selected_idx = np.stack(selected_idx)
        cloud_ind = np.stack(cloud_ind)

        selected_xyz = selected_pc[:, :, 0:3]
        selected_features = selected_pc[:, :, 3:7]

        flat_inputs = self.tf_map(selected_xyz, selected_features, selected_labels, selected_idx,
                                  cloud_ind)  

        num_layers = cfg.num_layers
        inputs = {}
        inputs['xyz'] = []
        for tmp in flat_inputs[:num_layers]:
            inputs['xyz'].append(torch.from_numpy(tmp).float())  
        inputs['neigh_idx'] = []
        for tmp in flat_inputs[num_layers: 2 * num_layers]:
            inputs['neigh_idx'].append(torch.from_numpy(tmp).long()) 
        inputs['sub_idx'] = []
        for tmp in flat_inputs[2 * num_layers:3 * num_layers]:
            inputs['sub_idx'].append(torch.from_numpy(tmp).long())  
        inputs['interp_idx'] = []
        for tmp in flat_inputs[3 * num_layers:4 * num_layers]:
            inputs['interp_idx'].append(torch.from_numpy(tmp).long())  

        # inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).transpose(1,2).float()   
        inputs['features'] = torch.from_numpy(flat_inputs[4 * num_layers]).float()  
        inputs['labels'] = torch.from_numpy(flat_inputs[4 * num_layers + 1]).long()
        inputs['input_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 2]).long()
        inputs['cloud_inds'] = torch.from_numpy(flat_inputs[4 * num_layers + 3]).long()

        return inputs


if __name__ == '__main__':  # use to test
    dataset = HongKong(val_split='Track_B_6')
    dataset_train = HongKongSampler(dataset, split='training')
    dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True,
                            collate_fn=dataset_train.collate_fn)
    # dataloader = DataLoader(dataset_train, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    for data in dataloader:
        features = data['features']
        labels = data['labels']
        idx = data['input_inds']
        cloud_idx = data['cloud_inds']
        print(features.shape)
        print(labels.shape)
        print(idx.shape)
        print(cloud_idx.shape)

        # change features into a numpy array, and then use open3d to visualize
        # conver features into a 2-D numpy array
        features = features.numpy()

        import open3d as o3d
        # show the first point cloud
        for i in range(features.shape[0]):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(features[i, :, 0:3])
            pcd.colors = o3d.utility.Vector3dVector(features[i, :, 3:6] / 255.0)
            # vis pcd
            o3d.visualization.draw_geometries([pcd])

        break





