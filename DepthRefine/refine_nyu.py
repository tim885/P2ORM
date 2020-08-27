import argparse
import os
import numpy as np
from tqdm import tqdm

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from lib.models.unet import UNet
from lib.utils.net_utils import load_checkpoint
from lib.utils.data_utils import padding_array, read_jiao, read_laina, read_sharpnet, read_eigen, read_dorn, read_bts, read_vnl


# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

# network settings
parser.add_argument('--checkpoint', type=str, default=None, help='optional reload model path')
parser.add_argument('--thresh', type=float, default=0.7, help='threshold value used to remove unconfident occlusions')
parser.add_argument('--use_occ', type=bool, default=True, help='whether to use occlusion as network input')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')

# pth settings
parser.add_argument('--result_dir', type=str, default='result/nyu')
parser.add_argument('--occ_dir', type=str, default='../data/NYUv2_OR/pred_occ')
parser.add_argument('--depth_dir', type=str, default='../data/NYUv2_OR/pred_depth')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# ================CREATE NETWORK AND OPTIMIZER============== #
net = UNet(use_occ=opt.use_occ)
optimizer = optim.Adam(net.parameters(), lr=opt.lr)

load_checkpoint(net, optimizer, opt.checkpoint)
net.cuda()
net.eval()
# ========================================================== #


# load in occlusion list
occ_list = sorted([name for name in os.listdir(opt.occ_dir) if name.endswith(".npz")])

# set the crop window defined in NYUv2 dataset
eigen_crop = [21, 461, 25, 617]


for method in tqdm(['jiao', 'laina', 'sharpnet', 'eigen', 'dorn', 'bts', 'vnl']):
    # create result dir
    result_dir = os.path.join(opt.result_dir, method)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # read in depths
    func = eval('read_{}'.format(method))
    depths_ori = func(opt.depth_dir)
    # padding the cropped depth prediction to 480x640
    depths = np.zeros((depths_ori.shape[0], 480, 640))
    depths[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]] = depths_ori
    depths = torch.from_numpy(np.ascontiguousarray(depths)).float().unsqueeze(1)
    assert len(occ_list) == depths.shape[0], 'depth map and occlusion map does not match !'

    with torch.no_grad():
        for i in tqdm(range(len(occ_list)), desc='refining depth prediction from {}'.format(method)):
            # load initial depth map predicted by Monocular Depth Estimation methods
            depth_coarse = depths[i].unsqueeze(0).cuda()

            # load predicted occlusion map
            occlusion = np.load(os.path.join(opt.occ_dir, occ_list[i]))['order']

            # remove predictions with small score
            mask = occlusion[:, :, 0] <= opt.thresh * 127
            occlusion[mask, 1:] = 0
            occlusion = padding_array(occlusion).unsqueeze(0).cuda()
            
            # forward pass
            depth_refined = net(depth_coarse, occlusion).clamp(1e-9).squeeze().cpu().numpy()
            depth_init = depth_coarse.squeeze().cpu().numpy()

            img_name = occ_list[i].split('-')[0]

            # save npy files
            np.save(os.path.join(result_dir, '{}_refine.npy'.format(img_name)), depth_refined)

            # visualization
            max_value = max(depth_refined.max(), depth_init.max())
            plt.imsave(os.path.join(result_dir, '{}_refine.png'.format(img_name)), depth_refined, vmin=0, vmax=max_value)
            plt.imsave(os.path.join(result_dir, '{}_init.png'.format(img_name)), depth_init, vmin=0, vmax=max_value)
