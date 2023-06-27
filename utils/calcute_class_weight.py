# --*-- UTF-8- --*--
# Data  :  上午9:54
# Name  : calcute_class_weight.py

import numpy as np
from pathlib import Path

import yaml

from utils.helper_ply import read_ply


class ClassWeight:

    def __init__(self, config_path, dataset_name, remap_label=False):
        self.config = config_path
        self.name = dataset_name
        self.remap_label = remap_label
        self.ignored_label = np.sort([])
        self.num_classes = None

        self.get_file_list()
        self.pre_class_num = self.get_per_class_num()

    def get_file_list(self):
        if self.name == 'HongKong':
            self.dataset_path = 'DATASET_PATH'
            self.data_list = list(Path(self.dataset_path).glob('*.ply'))
        else:
            raise ValueError('dataset name is not correct')

    def get_per_class_num(self):
        with open(self.config, 'r') as f:
            self.DATA = yaml.safe_load(f)
        if self.remap_label:
            self.label_to_names = self.DATA['new_labels']
        else:
            self.label_to_names = self.DATA['labels']

        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_names = [self.label_to_names[k] for k in self.label_values]
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.name_to_label = {v: k for k, v in self.label_to_names.items()}

        if self.num_classes is None:
            self.num_classes = len(self.label_to_names)

        num_per_class = [0 for _ in range(self.num_classes)]
        for file in self.data_list:
            data = read_ply(file)
            labels = data['class'].reshape(-1)

            if self.remap_label:
                labels = self.get_remap_label(labels)
            labels = np.array([self.label_to_idx[l] for l in labels])
            inds, counts = np.unique(labels, return_counts=True)
            for i, c in zip(inds, counts):
                if i == self.ignored_label:
                    continue
                else:
                    num_per_class[i] += c
        num_per_class = np.array(num_per_class)

        return num_per_class

    def get_remap_label(self, label):
        remap_dict = self.DATA['learning_map']
        for i, value in enumerate(remap_dict):
            label[label == value] = remap_dict[value]

        return label


if __name__ == '__main__':
    class_weight = ClassWeight(config_path='semantic-Vienna.yaml', dataset_name='Vienna', remap_label=True)
    print(class_weight.pre_class_num)
