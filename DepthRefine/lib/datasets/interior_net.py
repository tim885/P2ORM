import os
from os.path import join
import cv2
import pickle
import torch
import numpy as np
import pandas as pd

import torch.utils.data as data


class InteriorNet(data.Dataset):
    def __init__(self, root_dir, label_name='_raycastingV2',
                 pred_dir='pred', method_name='sharpnet_pred',
                 gt_dir='data', depth_ext='-depth-plane.png', normal_ext='-normal.png', im_ext='-rgb.png',
                 label_dir='label', label_ext='-order-pix.npy'):
        super(InteriorNet, self).__init__()
        self.root_dir = root_dir
        self.label_name = label_name
        self.method_name = method_name
        self.im_ext = im_ext
        self.gt_dir = gt_dir
        self.label_dir = label_dir
        self.pred_dir = pred_dir
        self.depth_ext = depth_ext
        self.normal_ext = normal_ext
        self.label_ext = label_ext
        self.df = pd.read_csv(join(root_dir, 'InteriorNet.txt'))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        depth_gt, depth_pred, label, normal, img = self._fetch_data(index)

        depth_gt = torch.from_numpy(np.ascontiguousarray(depth_gt)).float().unsqueeze(0)
        depth_pred = torch.from_numpy(np.ascontiguousarray(depth_pred)).float().unsqueeze(0)
        label = torch.from_numpy(np.ascontiguousarray(label)).float().permute(2, 0, 1)
        normal = torch.from_numpy(np.ascontiguousarray(normal)).float().permute(2, 0, 1)
        img = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1)

        return depth_gt, depth_pred, label, normal, img

    def _fetch_data(self, index):
        # fetch predicted depth map in meters
        depth_pred_path = join(self.root_dir, self.pred_dir, self.df.iloc[index]['scene'],
                               self.method_name, 'data', '{}.pkl'.format(self.df.iloc[index]['image']))
        with open(depth_pred_path, 'rb') as f:
            depth_pred = pickle.load(f)

        # fetch ground truth depth map in meters
        depth_gt_path = join(self.root_dir, self.gt_dir, 
                             '{}{}'.format(self.df.iloc[index]['scene'], self.label_name),
                             '{:04d}{}'.format(self.df.iloc[index]['image'], self.depth_ext))
        if not os.path.exists(depth_gt_path):
            print(depth_gt_path)
        depth_gt = cv2.imread(depth_gt_path, -1) / 1000

        # fetch normal map in norm-1 vectors
        normal_path = join(self.root_dir, self.gt_dir,
                           '{}{}'.format(self.df.iloc[index]['scene'], self.label_name),
                           '{:04d}{}'.format(self.df.iloc[index]['image'], self.normal_ext))
        normal = cv2.imread(normal_path, -1) / (2 ** 16 - 1) * 2 - 1
        normal = normal[:, :, ::-1]

        # fetch rgb image
        image_path = join(self.root_dir, self.gt_dir,
                          '{}{}'.format(self.df.iloc[index]['scene'], self.label_name),
                          '{:04d}{}'.format(self.df.iloc[index]['image'], self.im_ext))
        img = cv2.imread(image_path, -1) / 255
        img = img[:, :, ::-1]

        # fetch occlusion orientation labels
        label_path = join(self.root_dir, self.label_dir, 
                          '{}{}'.format(self.df.iloc[index]['scene'], self.label_name),
                          '{:04d}{}'.format(self.df.iloc[index]['image'], self.label_ext))
        label = np.load(label_path)

        return depth_gt, depth_pred, label, normal, img


if __name__ == "__main__":
    root_dir = '/space_sdd/InteriorNet'
    dataset = InteriorNet(root_dir)
    print(len(dataset))

    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import sys
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    for i, data in tqdm(enumerate(test_loader)):
        if i == 0:
            print(data[0].shape, data[1].shape, data[2].shape, data[3].shape, data[4].shape)
            sys.exit()
