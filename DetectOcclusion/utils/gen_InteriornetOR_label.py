# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------
# gen interiornetOR dataset P2ORM labels from interiornet dataset depth maps and normal maps

import os
import random
from tqdm import tqdm
import numpy as np
import torch
import cv2
import sys
import matplotlib.pyplot as plt
from skimage import feature
plt.switch_backend('agg')  # use matplotlib without gui support
sys.path.append('..')
from lib.dataset.gen_label_methods import gen_occ_order, occ_order_to_edge, viz_occ_order, occ_order_connect8_to_ori, viz_delta_depth

# InteriorNet camera intrinsic for 640 x 480
K = np.array([[600, 0, 320],
              [0, 600, 240],
              [0,   0,   1]])

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'InteriorNet'
org_dataset_root = os.path.join(cur_dir, '..', 'data', dataset_name)
gen_dataset_abs_root = os.path.join(cur_dir, '..', 'data', 'dataset_syn', dataset_name)
gen_dataset_rel_root = 'data/dataset_syn/{}'.format(dataset_name)
gen_data_dir = os.path.join(gen_dataset_abs_root, 'data')
np.set_printoptions(precision=4)  # print prec


def gen_occ_labels(mode, check_order=False):
    """
    gen occlusion edge and P2ORM labels between neighbor pixels(3x3)
    :param mode: ['normal'|'debug']
    :param check_order: vis p2orm labels
    """
    print('dataset_out_dir:', gen_data_dir)
    if not os.path.exists(gen_data_dir): os.makedirs(gen_data_dir)
    scenes_root = org_dataset_root  # root for scene subdirs
    org_rgb_dir = 'cam0'
    org_depth_dir = 'depth0'
    org_normal_dir = 'normal0'
    org_label_dir = 'label0'
    mask_gt_type = 'instance'

    thr_depth = 25.  # thresholds: depth_diff(mm)
    ROI_sz = 3  # region of interest size in images
    version_n = 'raycasting_{}mm_debug'.format(int(thr_depth))  # version name
    if mode == 'normal':  # all scenes for dataset generation
        sel_scene_list = sorted([fn for fn in os.listdir(scenes_root)])
    elif mode == 'debug':  # one scene dir for debug
        # sel_scene_list = ['3FO4IDP1HR4R_Balcony']
        sel_scene_list = ['3FO4MMLCRTM9_Living_room']

    for scene_id, scene_name in enumerate(tqdm(sel_scene_list)):
        cur_in_scene_dir = os.path.join(scenes_root, scene_name)
        cur_in_rgb_dir = os.path.join(cur_in_scene_dir, org_rgb_dir, 'data')
        cur_in_depth_dir = os.path.join(cur_in_scene_dir, org_depth_dir, 'data')
        cur_in_normal_dir = os.path.join(cur_in_scene_dir, org_normal_dir, 'data')
        cur_in_label_dir = os.path.join(cur_in_scene_dir, org_label_dir, 'data')

        curr_out_data_dir = os.path.join(gen_data_dir, '{}_{}'.format(scene_name, version_n))
        if not os.path.exists(curr_out_data_dir): os.makedirs(curr_out_data_dir)
        print('write data to {}'.format(curr_out_data_dir))

        file_ind = [name.replace('.png', '') for name in os.listdir(cur_in_rgb_dir)]
        file_ind.sort(key=int)
        file_names = ['{}'.format(idx) for idx in file_ind]

        for idx, file_name in enumerate(tqdm(file_names)):
            in_rgb_path = os.path.join(cur_in_rgb_dir, '{}.png'.format(file_name))
            in_depth_path = os.path.join(cur_in_depth_dir, '{}.png'.format(file_name))
            in_normal_path = os.path.join(cur_in_normal_dir, '{}.png'.format(file_name))
            in_label_path = os.path.join(cur_in_label_dir, '{}_{}.png'.format(file_name, mask_gt_type))

            out_depth_vis_path     = os.path.join(curr_out_data_dir, '{:04d}-depth_vis.png'.format(idx))
            out_depth_path         = os.path.join(curr_out_data_dir, '{:04d}-depth.png'.format(idx))
            out_rgb_path           = os.path.join(curr_out_data_dir, '{:04d}-rgb.png'.format(idx))
            out_normal_path        = os.path.join(curr_out_data_dir, '{:04d}-normal.png'.format(idx))
            out_occ_edge_path      = os.path.join(curr_out_data_dir, '{:04d}-edge.png'.format(idx))
            out_occ_edge_fg_path   = os.path.join(curr_out_data_dir, '{:04d}-edge_fg.png'.format(idx))
            out_occ_ori_path       = os.path.join(curr_out_data_dir, '{:04d}-ori.png'.format(idx))
            out_pixwise_order_path = os.path.join(curr_out_data_dir, '{:04d}-order-pix.npz'.format(idx))
            out_delta_abs_depth_path = os.path.join(curr_out_data_dir, '{:04d}-delta_abs_depth_check.png'.format(idx))
            out_delta_adj_depth_path = os.path.join(curr_out_data_dir, '{:04d}-delta_adj_depth_check.png'.format(idx))

            rgb = cv2.imread(in_rgb_path, cv2.IMREAD_COLOR)
            depth_org = cv2.imread(in_depth_path, cv2.IMREAD_UNCHANGED)  # uint16
            normal_org = cv2.imread(in_normal_path, cv2.IMREAD_UNCHANGED)  # z,y,x; point-to-camera
            mask_gt = cv2.imread(in_label_path, cv2.IMREAD_UNCHANGED)  # instance-wise mask (not accurate)
            mask_invalid = np.zeros(rgb.shape[:2])  # all pixels are valid in interiornet

            depth = np.copy(depth_org).astype(np.float32)
            normal = normal_org[:, :, [2, 1, 0]]  # z,y,x => x,y,z

            # gen occlusion order for each pixel w.r.t. its neighbor pixels
            _, occ_order_pix, delta_abs_depth, delta_adj_depth, _ = gen_occ_order(K, depth, mask_gt, mask_invalid, ROI_sz,
                                                                                  thr_depth, normal, lbl_type='mask',
                                                                                  dataset='interiornet', fast_abs_diff=False)
            if check_order:  # vis occ order
                occ_order_vis_path = os.path.join(curr_out_data_dir, '{:04d}-order-check.png'.format(idx))
                viz_occ_order(rgb[:, :, [2, 1, 0]], occ_order_pix, occ_order_vis_path)
                # viz_delta_depth(rgb[:, :, [2, 1, 0]], delta_abs_depth, out_delta_abs_depth_path, thr_depth * 2)
                # viz_delta_depth(rgb[:, :, [2, 1, 0]], delta_adj_depth, out_delta_adj_depth_path, thr_depth * 2)

            # gen occlusion edge from occlusion order
            occ_edge_fg, occ_edge_bg, occ_edge = occ_order_to_edge(occ_order_pix, connectivity=8)

            # gen occlusion orientation from occlusion order
            occ_ori = occ_order_connect8_to_ori(occ_order_pix[:, :, 1:])
            occ_ori[occ_edge_fg == 0] = 0  # only preserve occ_ori on occ_edge_fg
            occ_ori_vis = ((occ_ori + 1) / 2. * 255).astype(np.uint8)  # H,W

            # vis depth
            depth[depth == 0] = np.inf  # interiornet, inf for unknown region
            depth_vis = (depth.min() / depth) * 255.0  # save inverse depth for better vis

            # save img, depth, normal, occ edge, occ ori, occ order
            cv2.imwrite(out_rgb_path, rgb)
            cv2.imwrite(out_depth_path, depth_org)
            plt.imsave(out_depth_vis_path, depth_vis, vmin=depth_vis.min(), vmax=depth_vis.max())
            cv2.imwrite(out_normal_path, normal_org)
            cv2.imwrite(out_occ_edge_path, (occ_edge * 255.0).astype('uint8'))
            cv2.imwrite(out_occ_edge_fg_path, (occ_edge_fg * 255.0).astype('uint8'))
            cv2.imwrite(out_occ_ori_path, occ_ori_vis)
            np.savez_compressed(out_pixwise_order_path, order=occ_order_pix.astype('int8'))


def gen_occ_image_set():
    """gen train/val set for InteriorNet-OR dataset"""
    version_n = 'raycasting_25mm_debug'
    image_set_dir = os.path.join(gen_dataset_abs_root, 'image_sets')
    print('dataset data dir:{}'.format(gen_data_dir))
    if not os.path.exists(gen_data_dir):
        print('dataset does not exist!')
        sys.exit(1)
    if not os.path.exists(image_set_dir):
        os.makedirs(image_set_dir)

    # randomly split train/val
    scene_list = sorted([fn for fn in os.listdir(gen_data_dir) if version_n in fn])
    train_scene_num = int(len(scene_list) * 0.9)  # train/val split
    train_scene_list = random.sample(scene_list, k=train_scene_num)
    val_scene_list = list(set(scene_list) - set(train_scene_list))
    sel_scene_dict = {'train': train_scene_list, 'val': val_scene_list}
    print('train scene num: {}'.format(train_scene_list.__len__()))
    print('val scene num: {}'.format(val_scene_list.__len__()))

    # gen dataset csv
    for set_name, sel_scene_list in sel_scene_dict.items():
        out_csv_path = os.path.join(image_set_dir, 'occ_order_{0}_{1}.txt'.format(version_n, set_name))

        with open(out_csv_path, 'w') as f:
            lines = '#image_path,edge_label_path,ori_label_path,order_label_path\n'
            for scene_idx, scene_name in enumerate(sel_scene_list):
                scene_data_abs_dir = os.path.join(gen_data_dir, scene_name)
                scene_data_rel_dir = os.path.join(gen_dataset_rel_root, 'data', scene_name)

                rgb_files = sorted([fn for fn in os.listdir(scene_data_abs_dir) if 'rgb.png' in fn])
                occ_edge_files = sorted([fn for fn in os.listdir(scene_data_abs_dir) if 'edge_fg.png' in fn])
                occ_ori_files = sorted([fn for fn in os.listdir(scene_data_abs_dir) if 'ori.png' in fn])
                occ_order_files = sorted([fn for fn in os.listdir(scene_data_abs_dir) if 'order-pix.npz' in fn])

                for sample_idx, _ in enumerate(rgb_files):
                    rgb_file = os.path.join(scene_data_rel_dir, rgb_files[sample_idx])
                    edge_file = os.path.join(scene_data_rel_dir, occ_edge_files[sample_idx])
                    ori_file = os.path.join(scene_data_rel_dir, occ_ori_files[sample_idx])
                    order_file = os.path.join(scene_data_rel_dir, occ_order_files[sample_idx])
                    lines += '{0},{1},{2},{3}\n'.format(rgb_file, edge_file, ori_file, order_file)

            f.writelines(lines[:-1])  # rm last \n
            print('write to csv file:{}'.format(out_csv_path))


if __name__ == '__main__':
    # generate p2orm labels from depth and surface normal
    gen_occ_labels(mode='debug', check_order=False)

    # generate image set csv file for train/val
    # gen_occ_image_set()

