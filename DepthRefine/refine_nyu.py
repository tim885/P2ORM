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
parser.add_argument('checkpoint', type=str, default=None, help='optional reload model path')

parser.add_argument('--use_normal', action='store_true', help='whether to use normal map as network input')
parser.add_argument('--use_img', action='store_true', help='whether to use rgb image as network input')
parser.add_argument('--only_contour', action='store_true', help='whether to keep only the first channel of occlusion')
parser.add_argument('--use_occ', default=True, type=bool, help='whether to use occlusion as network input')
parser.add_argument('--no_contour', default=True, type=bool, help='whether to remove the first channel of occlusion')

parser.add_argument('--th', type=float, default=0.7)
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')

# pth settings
parser.add_argument('--result_dir', type=str, default='/space_sdd/NYU/depth_refine')
parser.add_argument('--occ_dir', type=str, default='/space_sdd/NYU/test_order_nms_pred_pretrain')
parser.add_argument('--data_dir', type=str, default='/home/xuchong/Projects/occ_edge_order/data/dataset_real/NYUv2/data/val_occ_order_raycasting_woNormal_avgROI_1mm')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# =================CREATE DATASET=========================== #
eigen_crop = [21, 461, 25, 617]


def read_jiao():
    ours = []
    jiao_pred_path = '/space_sdd/NYU/depth_predictions/jiao_pred_mat/'
    for i in range(654):
        f = loadmat(jiao_pred_path + str(i+1) + '.mat')
        f = f['pred']
        f = f[eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
        ours.append(f)
    ours = np.array(ours)
    return ours


def read_laina():
    laina_pred = h5py.File('/space_sdd/NYU/depth_predictions/laina_predictions_NYUval.mat', 'r')['predictions']
    laina_pred = np.array(laina_pred).transpose((0, 2, 1))
    laina_pred = laina_pred[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
    return laina_pred


def read_sharpnet():
    with open('/space_sdd/NYU/depth_predictions/sharpnet_prediction.pkl', 'rb') as f:
        ours = pkl.load(f)
    ours = np.array(ours)
    ours = ours[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
    return ours


def read_eigen():
    ours = loadmat('/space_sdd/NYU/depth_predictions/eigen_nyud_depth_predictions.mat')
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
    list_dirs = open('/space_sdd/NYU/depth_predictions/NYUV2_DORN/list_dorn_order.txt', 'r').readlines()
    for line in list_dirs:
        line = line.strip()
        f = loadmat('/space_sdd/NYU/depth_predictions/NYUV2_DORN/NYUV2_DORN/' + line)
        pred = f['pred'][eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
        ours.append(pred)
    ours = np.array(ours)
    return ours


def read_bts():
    ours = []
    list_dirs = open('/space_sdd/NYU/depth_predictions/result_bts_nyu/pred_bts.txt', 'r').readlines()
    tmp_dict = dict()
    for line in list_dirs:
        line = line.strip()
        num_tmp = line.rfind('_')
        key = int(line[num_tmp+1:-4])
        f = cv2.imread('/space_sdd/NYU/depth_predictions/result_bts_nyu/raw/' + line, -1)
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
    ours = pkl.load(open('/space_sdd/NYU/depth_predictions/pred_VNL.pkl', 'rb'))
    ours = np.array(ours) * 10
    ours = ours[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
    return ours
# ========================================================== #


# ================CREATE NETWORK AND OPTIMIZER============== #
net = UNet(use_occ=opt.use_occ, no_contour=opt.no_contour, only_contour=opt.only_contour,
           use_aux=(opt.use_normal or opt.use_img))
optimizer = optim.Adam(net.parameters(), lr=opt.lr)

load_checkpoint(net, optimizer, opt.checkpoint)
net.cuda()
net.eval()
# ========================================================== #


# load in occlusion list
occ_list = sorted([name for name in os.listdir(opt.occ_dir) if name.endswith(".npy")])

# load in normal list
normal_list = sorted([name for name in os.listdir(opt.data_dir) if name.endswith("-normal.png")])

# load in rgb list
img_list = sorted([name for name in os.listdir(opt.data_dir) if name.endswith("-rgb.png")])


for method in tqdm(['jiao', 'laina', 'sharpnet', 'eigen', 'dorn', 'bts', 'vnl']):
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
            depth_coarse = depths[i].unsqueeze(0).cuda()
            
            occlusion = np.load(os.path.join(opt.occ_dir, occ_list[i]))

            # remove predictions with small score
            mask = occlusion[:, :, 0] <= opt.th
            occlusion[mask, 1:] = 0

            occlusion = padding_array(occlusion).unsqueeze(0).cuda()
            #occlusion = torch.from_numpy(np.ascontiguousarray(occlusion)).float().permute(2, 0, 1).unsqueeze(0).cuda()

            if opt.use_normal:
                aux = cv2.imread(os.path.join(opt.data_dir, normal_list[i]), -1) / (2 ** 16 - 1) * 2 - 1
            elif opt.use_img:
                aux = cv2.imread(os.path.join(opt.data_dir, img_list[i]), -1) / 255
            else:
                aux = None
            if aux is not None:
                aux = padding_array(aux).unsqueeze(0).cuda()
            
            # forward pass
            pred = net(depth_coarse, occlusion, aux)

            depth_refine = pred.clamp(1e-9).squeeze().cpu().numpy()
            depth_init = depth_coarse.squeeze().cpu().numpy()

            img_name = occ_list[i].split('-')[0]
            refine_name = os.path.join(opt.result_dir, method, 'depth_refine', '{}.png'.format(img_name))
            #init_name = os.path.join(opt.result_dir, method, 'depth_init', '{}.png'.format(img_name))
            save_name = os.path.join(opt.result_dir, method, 'depth_npy', '{}.npy'.format(img_name))

            if not os.path.isdir(os.path.dirname(refine_name)):
                os.makedirs(os.path.dirname(refine_name))
            if not os.path.isdir(os.path.dirname(init_name)):
                os.makedirs(os.path.dirname(init_name))
            if not os.path.isdir(os.path.dirname(save_name)):
                os.makedirs(os.path.dirname(save_name))

            plt.imsave(refine_name, depth_refine)
            plt.imsave(init_name, depth_init)

            np.save(save_name, depth_refine)
