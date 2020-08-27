import os
import cv2
import pickle
import torch
import numpy as np

import torch.utils.data as data


class InteriorNet(data.Dataset):
    def __init__(self, root_dir):
        super(InteriorNet, self).__init__()
        self.root_dir = root_dir
        self.scenes = os.listdir(root_dir)

    def __len__(self):
        return len(self.scenes) * 20  # 20 images per scene in InteriorNet

    def __getitem__(self, index):
        depth_gt, depth_pred, occlusion, normal, img = self._fetch_data(index)

        depth_gt = torch.from_numpy(np.ascontiguousarray(depth_gt)).float().unsqueeze(0)
        depth_pred = torch.from_numpy(np.ascontiguousarray(depth_pred)).float().unsqueeze(0)
        occlusion = torch.from_numpy(np.ascontiguousarray(occlusion)).float().permute(2, 0, 1)
        normal = torch.from_numpy(np.ascontiguousarray(normal)).float().permute(2, 0, 1)
        img = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1)

        return depth_gt, depth_pred, occlusion, normal, img

    def _fetch_data(self, index):
        scene_dir = os.path.join(self.root_dir, self.scenes[index // 20])
        image_id = index % 20

        # fetch predicted depth map (SharpNet) in meters
        with open(os.path.join(scene_dir, '{:04d}-sharpnet-pred.pkl'.format(image_id)), 'rb') as f:
            depth_pred = pickle.load(f)

        # fetch ground truth depth map in meters
        depth_gt = cv2.imread(os.path.join(scene_dir, '{:04d}-depth-plane.png'.format(image_id)), -1) / 1000

        # fetch normal map in norm-1 vectors
        normal = cv2.imread(os.path.join(scene_dir, '{:04d}-normal.png'.format(image_id)), -1) / (2 ** 16 - 1) * 2 - 1
        normal = normal[:, :, ::-1]

        # fetch rgb image
        img = cv2.imread(os.path.join(scene_dir, '{:04d}-rgb.png'.format(image_id)), -1) / 255
        img = img[:, :, ::-1]

        # fetch occlusion orientation labels
        occlusion = np.load(os.path.join(scene_dir, '{:04d}-order-pix.npz'.format(image_id)))['order'].astype('float16')

        return depth_gt, depth_pred, occlusion, normal, img


if __name__ == "__main__":
    root_dir = '/space_sdd/XiaoDatasets/InteriorNet_OR/'
    dataset = InteriorNet(root_dir)
    print(len(dataset))

    from tqdm import tqdm
    from torch.utils.data import DataLoader
    import sys
    test_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    for i, data in tqdm(enumerate(test_loader)):
        if i == 0:
            print(data[0].shape, data[1].shape, data[2].shape, data[3].shape, data[4].shape)
            sys.exit()
