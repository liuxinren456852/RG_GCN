import os
import sys
import glob
import h5py
import numpy as np
import torch
import copy
from torch.utils.data import Dataset

Data_dir = r'G:\ZZY\Toronto-3D'


g_classes = [x.rstrip() for x in open(os.path.join(Data_dir, 'toronto_3d_classes_8.txt'))]
g_class2label = {cls: i for i, cls in enumerate(g_classes)}
g_class2color = {'Ground': [0, 255, 0],
                 'Road_markings': [0, 0, 255],
                 'Natural': [0, 255, 255],
                 'Building': [255, 255, 0],
                 'Utility_line': [255, 0, 255],
                 'Pole': [100, 100, 255],
                 'Car': [200, 200, 100],
                 'Fence': [170, 120, 200]}
g_label2color = {g_classes.index(cls): g_class2color[cls] for cls in g_classes}


def load_Toronto3D(partition, test_area):
    data_dir = r'G:\ZZY\Toronto-3D\Toronto3D_h5'
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(data_dir, f), 'r+')
        data = file["data"][:]
        data_copy = copy.deepcopy(data)
        label = file["label"][:]
        data_batchlist.append(data_copy)
        label_batchlist.append(label)
    # print(type(label_batchlist))
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    if test_area != 'all':
        test_area_name = "00" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


class Toronto3D(Dataset):
    def __init__(self, num_points=512, partition='train', test_area='2'):
        self.data, self.seg = load_Toronto3D(partition, test_area)
        self.seg -= 1
        self.num_points = num_points
        self.partition = partition
        self.alpha_list = [0 for _ in range(8)]
        for i in range(8):
            self.alpha_list[i] = np.sum((self.seg == i))

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        seg = self.seg[item][:self.num_points]
        if self.partition == 'train':
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        seg = torch.LongTensor(seg)
        return pointcloud, seg

    def __len__(self):
        return self.data.shape[0]

    def get_alphalist(self):
        return self.alpha_list