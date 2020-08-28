import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
from tqdm import tqdm

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt

from lib.models.unet import UNet
from lib.datasets.ibims import Ibims

from lib.utils.net_utils import load_checkpoint
from lib.utils.evaluate_ibims_error_metrics import compute_global_errors, \
    compute_depth_boundary_error, compute_directed_depth_error

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

# network settings
parser.add_argument('--checkpoint', type=str, default=None, help='optional reload model path')
parser.add_argument('--thresh', type=float, default=0.7, help='threshold value used to remove unconfident occlusions')
parser.add_argument('--use_occ', type=bool, default=True, help='whether to use occlusion as network input')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')

# pth settings
parser.add_argument('--result_dir', type=str, default='result/ibims', help='result folder')
parser.add_argument('--data_dir', type=str, default='../data/iBims1_OR', help='testing dataset')
parser.add_argument('--depth_pred_method', type=str, default='junli')
parser.add_argument('--occ_dir', type=str, default='pred_occ')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# =================CREATE DATASET=========================== #
dataset_val = Ibims(opt.data_dir, opt.depth_pred_method, opt.occ_dir, th=opt.thresh)
val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False)

with open(os.path.join(opt.data_dir, 'imagelist.txt')) as f:
    image_names = f.readlines()
image_names = [x.strip() for x in image_names]
# ========================================================== #


# ================CREATE NETWORK AND OPTIMIZER============== #
net = UNet(use_occ=opt.use_occ)
optimizer = optim.Adam(net.parameters(), lr=opt.lr)

load_checkpoint(net, optimizer, opt.checkpoint)
net.cuda()
# ========================================================== #


# ===================== DEFINE TEST ======================== #
def test(data_loader, net, result_dir):
    # Initialize global and geometric errors ...
    num_samples = len(data_loader)
    rms = np.zeros(num_samples, np.float32)
    log10 = np.zeros(num_samples, np.float32)
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    thr1 = np.zeros(num_samples, np.float32)
    thr2 = np.zeros(num_samples, np.float32)
    thr3 = np.zeros(num_samples, np.float32)

    dbe_acc = np.zeros(num_samples, np.float32)
    dbe_com = np.zeros(num_samples, np.float32)

    dde_0 = np.zeros(num_samples, np.float32)
    dde_m = np.zeros(num_samples, np.float32)
    dde_p = np.zeros(num_samples, np.float32)

    net.eval()
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader)):
            # load data and label
            depth_gt, depth_coarse, occlusion, edge = data
            depth_gt, depth_coarse, occlusion = depth_gt.cuda(), depth_coarse.cuda(), occlusion.cuda()
            edge = edge.numpy()

            # forward pass
            depth_refined = net(depth_coarse, occlusion).clamp(1e-9)

            # mask out invalid depth values
            valid_mask = (depth_gt != 0).float()
            gt_valid = depth_gt * valid_mask
            refine_valid = depth_refined * valid_mask
            init_valid = depth_coarse * valid_mask

            # get numpy array from torch tensor
            gt = gt_valid.squeeze().cpu().numpy()
            refine = refine_valid.squeeze().cpu().numpy()
            init = init_valid.squeeze().cpu().numpy()

            # save npy files
            np.save(os.path.join(result_dir, '{}_refine.npy'.format(image_names[i])), refine)

            # visualization
            max_value = max(refine.max(), init.max())
            plt.imsave(os.path.join(result_dir, '{}_refine.png'.format(image_names[i])), refine, vmin=0, vmax=max_value)
            plt.imsave(os.path.join(result_dir, '{}_init.png'.format(image_names[i])), init, vmin=0, vmax=max_value)

            # compute metrics
            gt_vec = gt.flatten()
            refine_vec = refine.flatten()

            abs_rel[i], sq_rel[i], rms[i], log10[i], thr1[i], thr2[i], thr3[i] = compute_global_errors(gt_vec, refine_vec)
            dbe_acc[i], dbe_com[i], est_edges = compute_depth_boundary_error(edge, refine)
            dde_0[i], dde_m[i], dde_p[i] = compute_directed_depth_error(gt_vec, refine_vec, 3.0)

    return abs_rel, sq_rel, rms, log10, thr1, thr2, thr3, dbe_acc, dbe_com, dde_0, dde_m, dde_p
# ========================================================== #


# save refined depth predictions
result_dir = os.path.join(opt.result_dir, opt.depth_pred_method)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

abs_rel, sq_rel, rms, log10, thr1, thr2, thr3, dbe_acc, dbe_com, dde_0, dde_m, dde_p = test(val_loader, net, result_dir)
print('############ Global Error Metrics #################')
print('rel    = ',  np.nanmean(abs_rel))
print('log10  = ',  np.nanmean(log10))
print('rms    = ',  np.nanmean(rms))
print('thr1   = ',  np.nanmean(thr1))
print('thr2   = ',  np.nanmean(thr2))
print('thr3   = ',  np.nanmean(thr3))
print('############ Depth Boundary Error Metrics #################')
print('dbe_acc = ',  np.nanmean(dbe_acc))
print('dbe_com = ',  np.nanmean(dbe_com))
print('############ Directed Depth Error Metrics #################')
print('dde_0  = ',  np.nanmean(dde_0)*100.)
print('dde_m  = ',  np.nanmean(dde_m)*100.)
print('dde_p  = ',  np.nanmean(dde_p)*100.)
