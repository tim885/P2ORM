import os
from os.path import join
import torch
import numpy as np
from scipy import io

import torch.utils.data as data


class Ibims(data.Dataset):
    def __init__(self, root_dir, method_name, occ_dir, th=None):
        super(Ibims, self).__init__()
        self.root_dir = root_dir
        self.occ_dir = occ_dir
        self.method_name = method_name
        self.th = th
        with open(join(self.root_dir, 'imagelist.txt')) as f:
            image_names = f.readlines()
        self.im_names = [x.strip() for x in image_names]

    def __len__(self):
        return len(self.im_names)

    def __getitem__(self, index):
        depth_gt, depth_pred, occ, edge = self._fetch_data(index)

        depth_gt = torch.from_numpy(np.ascontiguousarray(depth_gt)).float().unsqueeze(0)
        depth_pred = torch.from_numpy(np.ascontiguousarray(depth_pred)).float().unsqueeze(0)
        occ = torch.from_numpy(np.ascontiguousarray(occ)).float().permute(2, 0, 1)

        return depth_gt, depth_pred, occ, edge

    def _fetch_data(self, index):
        # fetch depth map in meters
        depth_gt_mat = join(self.root_dir, 'gt_depth', '{}.mat'.format(self.im_names[index]))
        depth_pred_mat = join(self.root_dir, 'pred_depth', self.method_name,
                              '{}_predictions_{}_results.mat'.format(self.im_names[index], self.method_name))
        depth_gt, depth_pred, edge = self._load_depths_from_mat(depth_gt_mat, depth_pred_mat)

        # fetch occlusion orientation maps
        occ_path = join(self.root_dir, self.occ_dir, self.im_names[index] + '-rgb-order-pix.npz')
        occ = np.load(occ_path)['order']

        # remove predictions with small score
        if self.th is not None:
            mask = occ[:, :, 0] <= self.th * 127
            occ[mask, 1:] = 0

        return depth_gt, depth_pred, occ, edge

    def _load_depths_from_mat(self, gt_mat, pred_mat):
        # load prediction depth
        pred = io.loadmat(pred_mat)['pred_depths']
        pred[np.isnan(pred)] = 0
        pred_invalid = pred.copy()
        pred_invalid[pred_invalid != 0] = 1

        # load ground truth depth
        image_data = io.loadmat(gt_mat)
        data = image_data['data']

        # extract neccessary data
        depth = data['depth'][0][0]  # Raw depth map
        mask_invalid = data['mask_invalid'][0][0]  # Mask for invalid pixels
        mask_transp = data['mask_transp'][0][0]  # Mask for transparent pixels
        edge = data['edges'][0][0]

        mask_missing = depth.copy()  # Mask for further missing depth values in depth map
        mask_missing[mask_missing != 0] = 1

        mask_valid = mask_transp * mask_invalid * mask_missing * pred_invalid  # Combine masks

        depth_valid = depth * mask_valid
        pred_valid = pred * mask_valid
        return depth_valid, pred_valid, edge
