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
from lib.dataset.gen_label_methods import gen_occ_order, occ_order_to_edge, viz_occ_order, \
    occ_order_connect4_to_occ_ori, occ_order_connect8_to_ori, viz_delta_depth

# InteriorNet camera intrinsic for 640 x 480
K = np.array([[600, 0, 320],
              [0, 600, 240],
              [0,   0,   1]])

cur_dir = os.path.dirname(os.path.abspath(__file__))
dataset_name = 'InteriorNet'
data_org_dir = os.path.join(cur_dir, '..', 'data', dataset_name)
dataset_out_abs_dir = os.path.join(cur_dir, '..', 'data', 'dataset_syn', dataset_name)
dataset_out_rel_dir = 'data/dataset_syn/{}'.format(dataset_name)
data_out_dir = os.path.join(cur_dir, '..', 'data', 'dataset_syn', dataset_name, 'data')
label_out_dir = os.path.join(cur_dir, '..', 'data', 'dataset_syn', dataset_name, 'label')
np.set_printoptions(precision=4)  # print prec


def gen_occ_labels(check_order=True):
    """gen occlusion edge and P2ORM labels between neighbor pixels(3x3)"""
    print('dataset_out_dir:', data_out_dir)
    if not os.path.exists(data_out_dir): os.makedirs(data_out_dir)
    scenes_root = os.path.join(data_org_dir, 'HD7', 'selected')  # root for scene subdirs
    org_rgb_dir = 'cam0'
    org_depth_dir = 'depth0'
    org_normal_dir = 'normal0'
    org_label_dir = 'label0'
    mask_gt_type = 'instance'

    thr_depth = 25.  # thresholds: depth_diff(mm)
    ROI_sz = 3  # region of interest size in images
    version_n = 'raycasting_{}mm_debug'.format(int(thr_depth))  # version name
    mode = 'debug'  # ['debug'|'normal']

    if mode == 'normal':  # all scenes for dataset generation
        sel_scene_list = sorted([fn for fn in os.listdir(scenes_root)])
    elif mode == 'debug':  # one scene dir for debug
        sel_scene_list = ['3FO4MMLCRTM9_Living_room']

    for scene_id, scene_name in enumerate(tqdm(sel_scene_list)):
        cur_in_scene_dir = os.path.join(scenes_root, scene_name)
        cur_in_rgb_dir = os.path.join(cur_in_scene_dir, org_rgb_dir, 'data')
        cur_in_depth_dir = os.path.join(cur_in_scene_dir, org_depth_dir, 'data')
        cur_in_normal_dir = os.path.join(cur_in_scene_dir, org_normal_dir, 'data')
        cur_in_label_dir = os.path.join(cur_in_scene_dir, org_label_dir, 'data')

        curr_out_data_dir = os.path.join(data_out_dir, '{}_{}'.format(scene_name, version_n))
        curr_out_label_dir = os.path.join(label_out_dir, '{}_{}'.format(scene_name, version_n))
        if not os.path.exists(curr_out_data_dir): os.makedirs(curr_out_data_dir)
        if not os.path.exists(curr_out_label_dir): os.makedirs(curr_out_label_dir)
        print('write data to {}\nwrite label to {}'.format(curr_out_data_dir, curr_out_label_dir))

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
            normal_out_path        = os.path.join(curr_out_data_dir, '{:04d}-normal.png'.format(idx))
            occ_edge_out_path      = os.path.join(curr_out_data_dir, '{:04d}-edge.png'.format(idx))
            occ_edge_fg_out_path   = os.path.join(curr_out_data_dir, '{:04d}-edge_fg.png'.format(idx))
            occ_fgbg_out_path      = os.path.join(curr_out_data_dir, '{:04d}-edge_fgbg.png'.format(idx))
            occ_ori_out_path       = os.path.join(curr_out_data_dir, '{:04d}-ori.png'.format(idx))
            pixwise_order_out_path = os.path.join(curr_out_label_dir, '{:04d}-order-pix.npy'.format(idx))
            edgewise_order_out_path  = os.path.join(curr_out_label_dir, '{:04d}-order-edge.npy'.format(idx))
            delta_abs_depth_out_path = os.path.join(curr_out_data_dir, '{:04d}-delta_abs_depth_check.png'.format(idx))
            delta_adj_depth_out_path = os.path.join(curr_out_data_dir, '{:04d}-delta_adj_depth_check.png'.format(idx))

            rgb = cv2.imread(in_rgb_path, cv2.IMREAD_COLOR)
            depth_org = cv2.imread(in_depth_path, cv2.IMREAD_UNCHANGED)  # uint16
            normal_org = cv2.imread(in_normal_path, cv2.IMREAD_UNCHANGED)  # z,y,x; point-to-camera
            mask_gt = cv2.imread(in_label_path, cv2.IMREAD_UNCHANGED)  # instance-wise mask(not accurate )
            mask_invalid = np.zeros(rgb.shape[:2])  # all pixels are valid

            depth = np.copy(depth_org).astype(np.float32)  # np pass by ref.
            normal = normal_org[:, :, [2, 1, 0]]  # z,y,x => x,y,z
            # mask_edge = find_boundaries(mask_gt, mode='outer', background=0).astype(np.uint8)  # scipy way

            # find occlusion edge/order
            occ_edge_pix, occ_order_pix, delta_abs_depth, delta_adj_depth, _ = \
                gen_occ_order(K, depth, mask_gt, mask_invalid, ROI_sz, thr_depth, normal,
                              lbl_type='mask', dataset='interiornet', fast_abs_diff=False)

            # occ_order_edgewise = occ_pixwise2edgewise(occ_order_pix, edge_types=2)
            occ_edge_fg, occ_edge_bg, occ_edge = occ_order_to_edge(occ_order_pix, connectivity=8)
            if check_order:
                occ_order_vis_path = os.path.join(curr_out_data_dir, '{:04d}-order-check.png'.format(idx))
                viz_occ_order(rgb[:, :, [2, 1, 0]], occ_order_pix, occ_order_vis_path)
                viz_delta_depth(rgb[:, :, [2, 1, 0]], delta_abs_depth, delta_abs_depth_out_path, thr_depth * 2)
                viz_delta_depth(rgb[:, :, [2, 1, 0]], delta_adj_depth, delta_adj_depth_out_path, thr_depth * 2)

            depth[depth == 0] = np.inf  # interiornet, inf for unknown region
            depth = (depth.min() / depth) * 255.0  # save inverse depth for better vis

            occ_edge_fgbg = np.ones(occ_edge.shape) * 0.5  # fg/bg on occ edge
            occ_edge_fgbg[occ_edge_fg == 1] = 1.
            occ_edge_fgbg[occ_edge_bg == 1] = 0.

            cv2.imwrite(out_rgb_path, rgb)
            cv2.imwrite(out_depth_path, depth_org)
            plt.imsave(out_depth_vis_path, depth, vmin=depth.min(), vmax=depth.max())
            cv2.imwrite(occ_edge_out_path, (occ_edge * 255.0).astype('uint8'))
            cv2.imwrite(occ_edge_fg_out_path, (occ_edge_fg * 255.0).astype('uint8'))
            cv2.imwrite(occ_fgbg_out_path, (occ_edge_fgbg * 255.0).astype('uint8'))
            np.save(pixwise_order_out_path, occ_order_pix.astype('int16'))  # pix-wise label
            # np.save(edgewise_order_out_path, occ_order_edgewise.astype('int16'))  # edge-wise label

            # gen occ ori from occ order
            occ_order_ind = occ_order_pix + 1  # H,W,9 ; [-1,0,1] => [0,1,2]
            occ_order_h = torch.from_numpy(occ_order_ind[:, :, 5]).unsqueeze(0).unsqueeze(0)
            occ_order_v = torch.from_numpy(occ_order_ind[:, :, 7]).unsqueeze(0).unsqueeze(0)
            occ_ori = occ_order_connect4_to_occ_ori(torch.cat([occ_order_h, occ_order_v], dim=1))  # 1,1,H,W ; [-1, 1]
            occ_ori_img = np.array(((occ_ori + 1) / 2. * 255).squeeze()).astype(np.uint8)  # H,W
            cv2.imwrite(occ_ori_out_path, occ_ori_img)


def occ_order2occ_ori():
    """occ order to occ ori annotation"""
    print('occ order to occ ori')
    print('data_out_dir:', data_out_dir)

    version = 'raycastingV3'

    scenes_lbl_dir  = label_out_dir
    scenes_data_dir = data_out_dir
    scene_list = sorted([fn for fn in os.listdir(scenes_data_dir) if version in fn])
    sel_scene_list = scene_list
    # sel_scene_list = ['3FO4MKUOI2RR_Living_room_raycastingV2']

    for scene_id, scene_name in enumerate(tqdm(sel_scene_list)):
        print('curr scene:', scene_name)
        curr_scene_lbl_dir  = os.path.join(scenes_lbl_dir, scene_name)
        curr_scene_data_dir = os.path.join(scenes_data_dir, scene_name)

        file_ind = [int(name.replace('-order-pix.npy', ''))
                    for name in os.listdir(curr_scene_lbl_dir) if '-pix' in name]
        file_ind.sort(key=int)
        lbl_order_files = ['{:04d}-order-pix.npy'.format(idx) for idx in file_ind]

        for idx, lbl_order_fn in enumerate(tqdm(lbl_order_files)):
            lbl_edge_path    = os.path.join(curr_scene_data_dir, '{:04d}-edge_fg.png'.format(idx))
            lbl_order_path   = os.path.join(curr_scene_lbl_dir, lbl_order_fn)
            lbl_ori_out_path = os.path.join(curr_scene_data_dir, '{:04d}-ori.png'.format(idx))

            occ_edge_fg = cv2.imread(lbl_edge_path, cv2.IMREAD_UNCHANGED) / 255  # [0,255] => [0,1]
            occ_order_9 = (np.load(lbl_order_path, allow_pickle=True) + 1).astype(np.int8)   # H,W,9 ; [0,1,2]

            # occ_order_h = torch.from_numpy(occ_order_9[:, :, 5]).unsqueeze(0).unsqueeze(0)
            # occ_order_v = torch.from_numpy(occ_order_9[:, :, 7]).unsqueeze(0).unsqueeze(0)
            # occ_ori = occ_order_to_ori_edgewise(torch.cat([occ_order_h, occ_order_v], dim=1))  # 1,1,H,W ; [-1, 1]
            # occ_ori = np.array(occ_ori.squeeze())  # => H,W

            occ_order_8 = occ_order_9[:, :, 1:] - 1
            occ_ori     = occ_order_connect8_to_ori(occ_order_8)

            occ_ori[occ_edge_fg == 0] = 0  # only preserve edge_fg ori

            occ_ori_img = ((occ_ori + 1) / 2. * 255).astype(np.uint8)  # H,W
            cv2.imwrite(lbl_ori_out_path, occ_ori_img)


def gen_occ_image_set():
    """gen occlusion order input/target pair list"""
    version = 'raycastingV2'
    image_set_dir = os.path.join(dataset_out_abs_dir, 'image_sets')
    print('dataset data dir:{}'.format(data_out_dir))

    if not os.path.exists(data_out_dir):
        print('dataset does not exist!')
        sys.exit(1)

    if not os.path.exists(image_set_dir): os.makedirs(image_set_dir)

    scene_list = [fn for fn in os.listdir(data_out_dir) if version in fn]
    scene_list = sorted(scene_list)

    # randomly split train/val
    train_scene_num = int(len(scene_list) * 0.9)  # train/val split
    train_scene_list = random.sample(scene_list, k=train_scene_num)
    val_scene_list = list(set(scene_list) - set(train_scene_list))
    sel_scene_dict = {'train': train_scene_list, 'val': val_scene_list}
    print('train scene num: {}'.format(train_scene_list.__len__()))
    print('val scene num: {}'.format(val_scene_list.__len__()))

    # gen dataset csv
    for set_name, sel_scene_list in sel_scene_dict.items():
        out_csv_path = os.path.join(image_set_dir, 'occ_order_{0}NYUCanny_{1}.txt'.format(version, set_name))

        with open(out_csv_path, 'w') as f:
            lines = '#image_path,edge_label_path,ori_label_path,order_label_path\n'

            for scene_idx, scene_name in enumerate(sel_scene_list):
                print(scene_name)
                scene_data_dir_abs = os.path.join(data_out_dir, scene_name)
                scene_label_dir_abs = os.path.join(label_out_dir, scene_name)

                scene_data_dir = os.path.join(dataset_out_rel_dir, 'data', scene_name)
                scene_label_dir = os.path.join(dataset_out_rel_dir, 'label', scene_name)

                rgb_files = [fn for fn in os.listdir(scene_data_dir_abs) if 'rgbS2T.png' in fn]
                occ_edge_files = [fn for fn in os.listdir(scene_data_dir_abs) if 'edge_canny.png' in fn]
                occ_ori_files = [fn for fn in os.listdir(scene_data_dir_abs) if 'ori.png' in fn]
                occ_order_files = [fn for fn in os.listdir(scene_label_dir_abs) if 'order-pix.npy' in fn]

                rgb_files = sorted(rgb_files)
                occ_edge_files = sorted(occ_edge_files)
                occ_ori_files = sorted(occ_ori_files)
                occ_order_files = sorted(occ_order_files)

                for sample_idx, _ in enumerate(rgb_files):
                    rgb_file = os.path.join(scene_data_dir, rgb_files[sample_idx])
                    edge_file = os.path.join(scene_data_dir, occ_edge_files[sample_idx])
                    ori_file = os.path.join(scene_data_dir, occ_ori_files[sample_idx])
                    order_file = os.path.join(scene_label_dir, occ_order_files[sample_idx])

                    lines += '{0},{1},{2},{3}\n'.format(rgb_file, edge_file, ori_file, order_file)

            f.writelines(lines[:-1])  # rm last \n
            print('write to csv file:{}'.format(out_csv_path))


if __name__ == '__main__':
    gen_occ_labels()
    occ_order2occ_ori()
    gen_occ_image_set()

