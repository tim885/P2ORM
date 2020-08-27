import argparse
import os
import numpy as np
import pickle as pkl
import h5py
from scipy.io import loadmat
from tqdm import tqdm
import cv2
import gc

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from lib.models.unet import UNet
from lib.utils.net_utils import load_checkpoint
from lib.utils.data_utils import padding_array

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

# network settings
parser.add_argument('--checkpoint', type=str, default=None, help='optional reload model path')

parser.add_argument('--use_occ', type=bool, default=True, help='whether to use occlusion as network input')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')

# pth settings
parser.add_argument('--result_dir', type=str, default='result/nyu')
parser.add_argument('--occ_dir', type=str, default='../data/NYUv2_OR/pred_occ')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# =================CREATE DATASET=========================== #
eigen_crop = [21, 461, 25, 617]


def read_jiao():
    ours = []
    jiao_pred_path = '../data/NYUv2_OR/pred_depth/jiao_pred_mat/'
    for i in range(654):
        f = loadmat(jiao_pred_path + str(i+1) + '.mat')
        f = f['pred']
        f = f[eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
        ours.append(f)
    ours = np.array(ours)
    return ours


def read_laina():
    laina_pred = h5py.File('../data/NYUv2_OR/pred_depth/laina_predictions_NYUval.mat', 'r')['predictions']
    laina_pred = np.array(laina_pred).transpose((0, 2, 1))
    laina_pred = laina_pred[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
    return laina_pred


def read_sharpnet():
    with open('../data/NYUv2_OR/pred_depth/sharpnet_prediction.pkl', 'rb') as f:
        ours = pkl.load(f)
    ours = np.array(ours)
    ours = ours[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
    return ours


def read_eigen():
    ours = loadmat('../data/NYUv2_OR/pred_depth/eigen_nyud_depth_predictions.mat')
    ours = ours['fine_predictions']
    ours = ours.transpose((2, 0, 1))
    out = []
    for line in ours:
        line = cv2.resize(line, (592, 440))
        out.append(line)
    out = np.array(out)
    return out


def read_dorn():
    ours = []
    list_dirs = open('../data/NYUv2_OR/pred_depth/NYUV2_DORN/list_dorn_order.txt', 'r').readlines()
    for line in list_dirs:
        line = line.strip()
        f = loadmat('../data/NYUv2_OR/pred_depth/NYUV2_DORN/NYUV2_DORN/' + line)
        pred = f['pred'][eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
        ours.append(pred)
    ours = np.array(ours)
    return ours


def read_bts():
    ours = []
    list_dirs = open('../data/NYUv2_OR/pred_depth/result_bts_nyu/pred_bts.txt', 'r').readlines()
    tmp_dict = dict()
    for line in list_dirs:
        line = line.strip()
        num_tmp = line.rfind('_')
        key = int(line[num_tmp+1:-4])
        f = cv2.imread('../data/NYUv2_OR/pred_depth/result_bts_nyu/raw/' + line, -1)
        f = f / 1000
        pred = f[eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
        tmp_dict[key] = pred
    keys = list(tmp_dict.keys())
    keys.sort()
    for key in keys:
        ours.append(tmp_dict[key])
    del tmp_dict
    gc.collect()
    ours = np.array(ours)
    return ours


def read_vnl():
    ours = pkl.load(open('../data/NYUv2_OR/pred_depth/pred_VNL.pkl', 'rb'))
    ours = np.array(ours) * 10
    ours = ours[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
    return ours
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


for method in tqdm(['jiao', 'laina', 'sharpnet', 'eigen', 'dorn', 'bts', 'vnl']):
    # create result dir
    result_dir = os.path.join(opt.result_dir, method)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # read in depths
    func = eval('read_{}'.format(method))
    depths_ori = func()
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
            occlusion = np.load(os.path.join(opt.occ_dir, occ_list[i]))['order'].astype('float')

            # remove predictions with small score
            mask = occlusion[:, :, 0] <= 0.5 * 128
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
