# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------
# gen pairwise occlusion relationship filtered by occ edge prob after nms and visualize if needed

import os
import cv2
import numpy as np
import sys
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # use matplotlib without gui support
sys.path.append('..')
from lib.dataset.gen_label_methods import viz_occ_order

curr_dir = os.path.abspath(os.path.dirname(__file__))
dataset = 'nyuv2'

res_root_dir = os.path.join(curr_dir, '..', 'experiments/output/NYUv2OR_pretrained/results_vis/test_74_{}'.format(dataset))
res_img_dir  = os.path.join(res_root_dir, 'images')
res_order_dir     = os.path.join(res_root_dir, 'res_mat', 'test_order_pred')
res_order_nms_dir = os.path.join(res_root_dir, 'res_mat', 'test_order_nms_pred')
if not os.path.exists(res_order_nms_dir): os.makedirs(res_order_nms_dir)

if dataset == 'ibims':
    testIds_path = os.path.join(curr_dir, '..', 'data/iBims1_OR/test_ori_iids.txt')
elif dataset == 'nyuv2':
    testIds_path = os.path.join(curr_dir, '..', 'data/NYUv2_OR/test_iids.txt')
elif dataset == 'BSDSownership':
    testIds_path = os.path.join(curr_dir, '..', 'data/BSDS300/BSDS_theta/test_ori_iids.txt')

with open(testIds_path) as f:
    test_ids = f.readlines() 

check_order = False  # viz occ order
prob_thresh = 0.5  # mask out occ edge region below threshold
for idx, big_idx in enumerate(tqdm(test_ids)):
    big_idx = big_idx.replace('\n', '')
    if dataset not in ['BSDSownership']: big_idx = big_idx + '-rgb'

    rgb_in_path    = os.path.join(res_img_dir, '{}_img_v.png'.format(big_idx))
    prob_in_path   = os.path.join(res_img_dir, '{}_lab_v_g_nms.png'.format(big_idx))
    order_in_path  = os.path.join(res_order_dir, '{}-order-pix.npy'.format(big_idx))
    order_out_path = os.path.join(res_order_nms_dir, '{}-order-pix.npz'.format(big_idx))
    viz_out_path   = os.path.join(res_order_nms_dir, '{}-order-viz.png'.format(big_idx))
    order_out_name = order_out_path.split('/')[-1]
    order_out_dir  = order_out_path.replace(order_out_name, '')
    if not os.path.exists(order_out_dir): os.makedirs(order_out_dir)

    prob_nms = cv2.imread(prob_in_path, cv2.IMREAD_UNCHANGED)  # uint8
    prob_nms = prob_nms.astype(np.float32) / 255  # [0~255] => [0~1]

    occ_order     = np.load(order_in_path, allow_pickle=True)  # occ_edge_prob + occ_order ;H,W,9; {-1,0,1}
    occ_order_nms = occ_order.copy()

    no_occ = (prob_nms <= prob_thresh)
    occ_order_nms[no_occ, 1:] = 0  # filter out non-occ region by thresholding
    occ_order_nms[:, :, 0] = prob_nms * 127  # [0~1] => [0~127]
    np.savez_compressed(order_out_path, order=occ_order_nms.astype(np.int8))

    if check_order:
        bgr = cv2.imread(rgb_in_path, cv2.IMREAD_UNCHANGED)  # uint8
        if bgr is None:
            H, W = prob_nms.shape
            bgr = np.zeros((H, W, 3))
        rgb = bgr[:, :, [2, 1, 0]]
        viz_occ_order(rgb, occ_order_nms, viz_out_path)  # occ order along 8 neighbor pixels

        # plot pairwise occlusion order along four inclinations
        cm = plt.get_cmap('bwr')
        occ_order_E  = (occ_order_nms[:, :, 5] + 1.) / 2  # 5,7,8,3  # {0,0.5,1}
        occ_order_S  = (occ_order_nms[:, :, 7] + 1.) / 2
        occ_order_SE = (occ_order_nms[:, :, 8] + 1.) / 2
        occ_order_NE = (occ_order_nms[:, :, 3] + 1.) / 2

        colored_image_E  = cm(occ_order_E)
        colored_image_S  = cm(occ_order_S)
        colored_image_SE = cm(occ_order_SE)
        colored_image_NE = cm(occ_order_NE)

        Image.fromarray((colored_image_E[:, :, :3] * 255).astype(np.uint8)).save(viz_out_path.replace('-viz', '-viz-E'))
        Image.fromarray((colored_image_S[:, :, :3] * 255).astype(np.uint8)).save(viz_out_path.replace('-viz', '-viz-S'))
        Image.fromarray((colored_image_SE[:, :, :3] * 255).astype(np.uint8)).save(viz_out_path.replace('-viz', '-viz-SE'))
        Image.fromarray((colored_image_NE[:, :, :3] * 255).astype(np.uint8)).save(viz_out_path.replace('-viz', '-viz-NE'))





