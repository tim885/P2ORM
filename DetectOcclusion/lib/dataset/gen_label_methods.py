# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------

import cv2
import sys
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import itertools
from PIL import Image
from scipy import pi, ndimage
sys.path.append('../..')
from math import atan, tan
PI = 3.1416


# ===================================== functions for dataset generation ============================================= #
def gen_occ_order(K, depth, label_map, invalid_mask, ROI_sz, thr_depth, normal=None, lbl_type='mask',
                  depth_avg=False, dataset='interiornet', thr_pix=False, fast_abs_diff=False):
    """
    convert depth to pixel-wise occ edge and pairwise occ order with givendepth , then corrected
    by normal map and instance mask edge(optional)
    :param K: current image camera intrinsic 
    :param depth: Euclidean distance between camera center and relevant pixel's 3D point
    :param label_map: instance mask or edge mask which indicates image edge
    :param invalid_mask: invalid raw data mask; [valid:0, invalid:1]
    :param ROI_sz: size of region to determine occlusion order, default=3
    :param lbl_type: ['edge'|'mask']: labeled occlusion edge or semantic mask
    :param thr_depth: neighbor pixels depth difference rate (depth_diff / pixel_dist) thresh to detect occlusion
    :param depth_avg: whether use average depth over one pixel's neighbor as pixel depth
    :param dataset: dataset name, for dataset-specific pre-processing
    :param thr_pix: whether use pixel-wise discontinuity threshold
    :return occ_label: [H, W, (1(edge) + 8(order))]
    """
    # pre-process depth, normal, label
    H, W = depth.shape
    padding = 2  # padding for depth
    depth_pad_2 = cv2.copyMakeBorder(depth, padding, padding, padding, padding, cv2.BORDER_REPLICATE)  # H+2,W+2
    invalid_mask_pad = cv2.copyMakeBorder(invalid_mask, padding, padding, padding, padding, cv2.BORDER_REPLICATE)  # H+2,W+2

    if normal is not None:
        if normal.dtype == np.uint16:  # [0,65535] => [-1, 1]
            normal = normal.astype(np.float32) / 65535.0 * 2 - 1.0  # H,W,3
            normal[:, :, 1] = -normal[:, :, 1]  # y-down => y-up, interiornet case
            normal[:, :, 2] = -normal[:, :, 2]  # z-in => z-out
        normal_pad = cv2.copyMakeBorder(normal, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    if lbl_type == 'edge':  # occ edge map
        edge_mask = label_map
        edge_mask_pad = cv2.copyMakeBorder(edge_mask, 1, 1, 1, 1, cv2.BORDER_REPLICATE)

    # init for ray-casting method
    occ_edge         = np.zeros(depth.shape[:2])  # 2-pix width occ edge (fg+bg)
    occ_label        = np.zeros((depth.shape[0], depth.shape[1], 9))  # occ edge + occ order (-1,0,1) w.r.t. 8 neighbors
    occ_label_tmp    = np.zeros((depth.shape[0], depth.shape[1], 9))
    diff_abs_depth   = np.zeros((depth.shape[0], depth.shape[1], 8))  # depth diff w.r.t 8 neighbors
    diff_adj_depth   = np.zeros((depth.shape[0], depth.shape[1], 8))  # adjusted depth diff (mid-pix ray) w.r.t 8 neighbors
    shifts_pix = [[-1, -1], [-1, 0], [-1, 1],
                  [0,  -1],          [0,  1],
                  [1,  -1], [1,  0], [1,  1]]
    shift_midpix = np.array([[-0.5, -0.5], [-0.5, 0.0], [-0.5, 0.5],
                             [0.0, -0.5],                [0.0, 0.5],
                             [0.5, -0.5],   [0.5, 0.0],  [0.5, 0.5]])  # shift from center pix to mid pix
    origin = np.zeros((8, 3))  # I0 of rays, namely camera optical center
    depth_err_map = np.zeros(depth.shape[:2])  # estimated GT depth error (only for real dataset)
    # thr_depth_const = thr_depth

    # firstly check absolute depth diff (avoid ROI op, check)
    depth_pad_1 = depth_pad_2[1:-1, 1:-1]  # H+1,W+1
    for idx, shift_pix in enumerate(shifts_pix):
        shift_h, shift_w = shift_pix
        pix_dist = 1.414 if idx in [0, 2, 5, 7] else 1.  # distance between neighbor pixels
        depth_diff = (depth_pad_1[1 + shift_h:H + 1 + shift_h, 1 + shift_w:W + 1 + shift_w] - depth) / pix_dist  # H,W
        diff_abs_depth[:, :, idx] = depth_diff
        occ_label_tmp[depth_diff > thr_depth, idx + 1] = 1.  # fg
        occ_label_tmp[depth_diff < -thr_depth, idx + 1] = -1.  # bg
    occ_exist_bool = np.any((occ_label_tmp != 0), axis=2)  # H,W

    if fast_abs_diff:  # fast mode using only absolute depth difference as ablation study
        occ_edge[occ_exist_bool] = 1.0
        occ_label = occ_label_tmp
        occ_label[occ_exist_bool, 0] = 1.0
        return occ_edge, occ_label, diff_abs_depth, diff_adj_depth, depth_err_map

    # gen occ order for each pixel over the image
    for y_idx in range(0, depth.shape[0]):
        for x_idx in range(0, depth.shape[1]):
            if invalid_mask[y_idx, x_idx] == 1: continue  # skip pixel
            if occ_exist_bool[y_idx, x_idx] != 1: continue
            ROI_depth_L   = np.copy(depth_pad_2[y_idx:(y_idx + ROI_sz + padding), x_idx:(x_idx + ROI_sz + padding)])
            ROI_invalid_L = np.copy(invalid_mask_pad[y_idx:(y_idx + ROI_sz + padding), x_idx:(x_idx + ROI_sz + padding)])

            # ============================= special pre-processing for dataset ======================================= #
            if dataset in ['interiornet', 'scenenet']:
                if ROI_depth_L.min() == 0.0:  # inf depth
                    ROI_depth_L[ROI_depth_L != 0.] = ROI_depth_L.max()  # rm depth edge problem
                    ROI_depth_L[ROI_depth_L == 0.] = 65535.0  # max depth for inf depth
            elif dataset == 'ibims':
                if ROI_depth_L[2, 2] == 0:
                    continue  # invalid center pixel, skip
                else:
                    if thr_pix:  # cal curr pixel depth discontinuity thresh
                        eta_d_ibims = 0.002  # depth angular err for ibims-1 dataset
                        err_d_ibims = 1.  # depth translational err for ibims-1 dataset

                        center_2D    = np.array([y_idx + 0.5, x_idx + 0.5], dtype=np.float32)  # 2,
                        neighbors_2D = center_2D + 2. * shift_midpix  # 8,2
                        ROI_2D       = np.insert(neighbors_2D, int((ROI_sz ** 2 - 1) / 2),
                                                 center_2D, axis=0).astype(np.float32)  # 9,2

                        center_ray      = np.array([center_2D[1] - K[0, 2], K[1, 2] - center_2D[0], -K[0, 0]])  # 3,
                        # center_ray_unit = center_ray / np.linalg.norm(center_ray)  # 3,
                        ROI_rays      = np.stack((ROI_2D[:, 1] - K[0, 2],
                                                 K[1, 2] - ROI_2D[:, 0],
                                                 -K[0, 0].repeat(9)), axis=1)  # 9,3
                        ROI_rays_unit = ROI_rays / np.linalg.norm(ROI_rays, axis=1).reshape(-1, 1)  # 9,3

                        ROI_normal         = np.copy(normal_pad[y_idx:(y_idx + ROI_sz), x_idx:(x_idx + ROI_sz), :]).reshape(-1, 3)  # 3,3,3 => 9,3
                        ROI_normal_unit    = ROI_normal / np.linalg.norm(ROI_normal, axis=1).reshape(-1, 1)  # 9,3
                        center_normal      = np.copy(normal_pad[y_idx+1, x_idx+1, :])  # 3,
                        center_normal_unit = center_normal / np.linalg.norm(center_normal)

                        # gazing angle between surface and line of sight
                        # gamma     = np.arccos(np.sum(center_ray_unit * center_normal_unit)) - PI / 2
                        gamma_roi = np.arccos(np.sum(ROI_rays_unit * ROI_normal_unit, axis=1)) - PI / 2  # 9,
                        # if np.any(gamma_roi <= eta_d_ibims): continue  # condition for depth err caused by angular err

                        tan_gamma = np.minimum(np.tan(gamma_roi), 1.)  # consider possible normal estimation err
                        tan_gamma = np.maximum(tan_gamma, 0.0001)  # required: tan(gamma) >> tan(err_d_ibims)
                        depth_err = eta_d_ibims / tan_gamma * ROI_depth_L[2, 2] + err_d_ibims
                        thr_depth = 25. + depth_err[4] + np.delete(depth_err, 4)  # 8,
                        depth_err_map[y_idx, x_idx] = depth_err[4]

                    # guess zero-value neighbor depth by 3x3 average depth
                    if np.any(ROI_depth_L[1:-1, 1:-1] == 0):
                        for y in range(0, ROI_sz):
                            for x in range(0, ROI_sz):
                                if ROI_depth_L[y+1, x+1] == 0.:
                                    ROI_depth_valid = ROI_depth_L[y:y + ROI_sz, x:x + ROI_sz]
                                    ROI_depth_valid = ROI_depth_valid[ROI_depth_valid != 0]
                                    ROI_depth_L[y+1, x+1] = ROI_depth_valid.mean()
            # ======================================================================================================== #

            ROI_depth = np.zeros((ROI_sz, ROI_sz))
            if depth_avg:  # avg each pixel depth in ROI
                for y in range(0, ROI_sz):
                    for x in range(0, ROI_sz):
                        ROI_depth[y, x] = np.mean(ROI_depth_L[y:y + ROI_sz, x:x + ROI_sz])
            else:
                ROI_depth   = ROI_depth_L[1:-1, 1:-1]  # 3x3
                ROI_invalid = ROI_invalid_L[1:-1, 1:-1]  # 3x3

            # pixel idx in flat vector and its relevant location in connectivity-8 neighborhood
            # 0 1 2
            # 3   4
            # 5 6 7
            center_depth         = ROI_depth[int((ROI_sz - 1) / 2), int((ROI_sz - 1) / 2)]
            ROI_depth_flat       = ROI_depth.flatten()
            neighbors_depth_flat = np.delete(ROI_depth_flat, (ROI_sz * ROI_sz - 1) / 2)  # 8,
            ROI_invalid_flat       = ROI_invalid.flatten()
            neighbors_invalid_flat = np.delete(ROI_invalid_flat, (ROI_sz * ROI_sz - 1) / 2)  # 8,

            ROI_depth_diff      = ROI_depth - center_depth  # cal abs depth diff
            ROI_depth_diff_flat = ROI_depth_diff.flatten()  # row-wise flatten
            ROI_depth_diff_flat = np.delete(ROI_depth_diff_flat, (ROI_sz * ROI_sz - 1) / 2)  # 8,
            ROI_depth_diff_flat[[0, 2, 5, 7]] = ROI_depth_diff_flat[[0, 2, 5, 7]] / 1.414  # for diagonal neighbors

            gen_occ_lbl = False
            if lbl_type == 'edge' and edge_mask[y_idx, x_idx] == 1:
                gen_occ_lbl = True
            elif lbl_type == 'mask' and np.any(np.abs(ROI_depth_diff).max() > thr_depth):
                gen_occ_lbl = True
            if gen_occ_lbl:  # gen occ edge/order
                # ======================= cal relevant discontinuities if normal is available ======================== #
                if normal is not None:
                    ROI_normal = np.copy(normal_pad[y_idx:(y_idx + ROI_sz), x_idx:(x_idx + ROI_sz), :]).reshape(-1, 3)  # 3,3,3 => 9,3
                    center_normal = ROI_normal[int((ROI_sz ** 2 - 1) / 2), :]  # 3,
                    neighbors_normal = np.delete(ROI_normal, int((ROI_sz ** 2 - 1) / 2), axis=0)  # 8,3

                    # gen relevant pixels coordinates on image plane
                    center_2D    = np.array([y_idx + 0.5, x_idx + 0.5], dtype=np.float32)
                    mid_2D       = center_2D + shift_midpix  # 8,2
                    neighbors_2D = center_2D + 2. * shift_midpix  # 8,2
                    ROI_2D       = np.insert(neighbors_2D, int((ROI_sz ** 2 - 1) / 2),
                                             center_2D, axis=0).astype(np.float32)  # 9,2

                    # gen rays from camera center to pixels/middle-pixels
                    mid_pix_rays = np.stack((mid_2D[:, 1] - K[0, 2],
                                             K[1, 2] - mid_2D[:, 0],
                                             -K[0, 0].repeat(8)), axis=1)  # 8,3
                    ROI_rays      = np.stack((ROI_2D[:, 1] - K[0, 2],
                                             K[1, 2] - ROI_2D[:, 0],
                                             -K[0, 0].repeat(9)), axis=1)  # 9,3
                    ROI_rays_unit = ROI_rays / np.linalg.norm(ROI_rays, axis=1).reshape(-1, 1)  # 9,3

                    # gen 3D points coordinates w.r.t. 2D ROI pixels
                    ROI_3D       = ROI_rays_unit * ROI_depth.reshape(-1, 1)  # 9,3
                    center_3D    = ROI_3D[int((ROI_sz ** 2 - 1) / 2), :]  # 3,
                    neighbors_3D = np.delete(ROI_3D, int((ROI_sz ** 2 - 1) / 2), axis=0)  # rm center; 8,3

                    # cal intersected points between mid pix rays and local tangent planes in 3D
                    pts_midray_centerplane    = insect_line_plane_3d_batch(origin, mid_pix_rays,
                                                                           center_3D, center_normal)  # 8,3
                    pts_midray_neighborplanes = insect_line_plane_3d_batch(origin, mid_pix_rays,
                                                                           neighbors_3D, neighbors_normal)  # 8,3

                    # ignore case where lines are parallel to planes
                    pts_midray_centerplane[np.isnan(pts_midray_centerplane)]       = 0.
                    pts_midray_neighborplanes[np.isnan(pts_midray_neighborplanes)] = 0.

                    # cal intersected points between center ray and neighbors local planes
                    pts_centerray_neighborplanes = insect_line_plane_3d_batch(origin, center_3D, neighbors_3D, neighbors_normal)

                    # cal intersected point between neighbors rays and center local plane
                    pts_neighborrays_centerplane = insect_line_plane_3d_batch(origin, neighbors_3D, center_3D, center_normal)

                    # ignore case where lines are parallel to plane
                    pts_centerray_neighborplanes[np.isnan(pts_centerray_neighborplanes)] = 0.  # 8,
                    pts_neighborrays_centerplane[np.isnan(pts_neighborrays_centerplane)] = 0.

                    dist_centerray_neighborplanes  = np.linalg.norm(pts_centerray_neighborplanes, axis=1)  # 8,
                    dist_neighborrays_centerplane  = np.linalg.norm(pts_neighborrays_centerplane, axis=1)  # 8,
                    diff_center2centerray_neighborplanes     = dist_centerray_neighborplanes - center_depth
                    diff_neighborrays_centerplane2neighbors  = neighbors_depth_flat - dist_neighborrays_centerplane

                    # gen depth diff through middle pixel
                    dist_midray_centerplane    = np.linalg.norm(pts_midray_centerplane, axis=1)  # 8,
                    dist_midray_neighborplanes = np.linalg.norm(pts_midray_neighborplanes, axis=1)  # 8,
                    diff_insects_center2neighbors = dist_midray_neighborplanes - dist_midray_centerplane  # 8,
                    diff_insects_center2neighbors[[0, 2, 5, 7]] = diff_insects_center2neighbors[[0, 2, 5, 7]] / 1.414  # for diagonal neighbors
                    diff_adj_depth[y_idx, x_idx, :] = diff_insects_center2neighbors  # 8,
                # ==================================================================================================== #

                # ======================================= gen occ order labels ======================================= #
                # gen occ order from zero-order depth diff (cond. 1)
                occ_order_flat = np.zeros((occ_label.shape[-1] - 1))  # occ order label vec; 8,
                occ_order_flat[ROI_depth_diff_flat > thr_depth]    = 1.  # center pixel is fg
                occ_order_flat[ROI_depth_diff_flat < (-thr_depth)] = -1.  # center pixel is bkg

                if normal is not None:
                    # check whether adjusted depth diff satisfy (cond. 2)
                    midray_check = np.zeros(int(ROI_sz ** 2 - 1))
                    midray_check[diff_insects_center2neighbors > thr_depth]    = 1.  # fg
                    midray_check[diff_insects_center2neighbors < (-thr_depth)] = -1.  # bg
                    consis_check_1 = midray_check * occ_order_flat  # [-1, 0, 1]
                    occ_order_flat[consis_check_1 != 1] = 0.

                    # check whether center pixel plane is always before/after neighbor pixel plane (cond. 3)
                    center_before_neighbor = np.zeros(8)
                    center_before_neighbor[(diff_center2centerray_neighborplanes > 0) * (diff_neighborrays_centerplane2neighbors > 0)] = 1
                    center_before_neighbor[(diff_center2centerray_neighborplanes < 0) * (diff_neighborrays_centerplane2neighbors < 0)] = -1
                    consis_check_2 = center_before_neighbor * occ_order_flat
                    occ_order_flat[consis_check_2 != 1] = 0.

                if lbl_type == 'edge':  # check whether neighbor pixel is also on occ edge
                    ROI_edge       = np.copy(edge_mask_pad[y_idx:(y_idx + ROI_sz),
                                                           x_idx:(x_idx + ROI_sz)]).flatten()  # 3,3 => 9,
                    neighbors_edge = np.delete(ROI_edge, int((ROI_sz ** 2 - 1) / 2), axis=0)  # 8,
                    occ_order_flat[neighbors_edge == 1] = 0.

                if np.abs(occ_order_flat.sum()) == 8:
                    occ_order_flat = np.zeros(8)  # discard case where center occlude/occluded by all neighbors

                occ_order_flat[neighbors_invalid_flat == 1] = 0  # remove all occ label with invalid neighbors
                occ_label[y_idx, x_idx, 1:] = occ_order_flat

                if lbl_type == 'edge':
                    occ_edge[y_idx, x_idx]     = 1.0
                    occ_label[y_idx, x_idx, 0] = 1.0
                elif lbl_type == 'mask':
                    if np.any(occ_order_flat != 0):  # all pixels where occ exists
                        occ_edge[y_idx, x_idx]     = 1.0
                        occ_label[y_idx, x_idx, 0] = 1.0
                # ==================================================================================================== #

    return occ_edge, occ_label, diff_abs_depth, diff_adj_depth, depth_err_map


def occ_fg_to_order(fg, bg, ROI_sz=3, fill_bg_hole=True, interpolate=False):
    """
    get pairwise occ order label from Figure/Ground label
    :param fg: numpy, [0,1], H,W
    :param bg: numpy, [0,-1] H,W
    :param interpolate: use connect-4 label to gen connect-8 label
    :return: occ_order_pair: H,W,9 ; new_occ_bg, new_occ_fg: refined fg/bg with holes filled
    """
    H, W = fg.shape
    bg[fg == 1] = 0  # rm wrongly annotated bg pixels

    bg_pad_1 = cv2.copyMakeBorder(bg, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    fg_pad_2 = cv2.copyMakeBorder(fg, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    bg_pad_2 = cv2.copyMakeBorder(bg, 2, 2, 2, 2, cv2.BORDER_REPLICATE)
    new_bg_pad_1 = np.zeros((H + 2, W + 2))
    new_fg_pad_1 = np.zeros((H + 2, W + 2))
    occ_order_pair_pad = np.zeros((H + 2, W + 2, 9))

    for y_idx in range(0, H):
        for x_idx in range(0, W):
            if fg[y_idx, x_idx] != 1: continue  # assume contour consists of fg pixs
            bg_ROI_L = np.copy(bg_pad_2[y_idx:(y_idx + 5), x_idx:(x_idx + 5)])  # 5,5
            fg_ROI_L = np.copy(fg_pad_2[y_idx:(y_idx + 5), x_idx:(x_idx + 5)])  # 5,5

            # fill holes in bg_ROI_L which should be bg
            if fill_bg_hole:
                bg_ROI_L[bg_ROI_L == -1] = 1
                bg_ROI_L[fg_ROI_L == 1] = 1  # make binary img
                bg_ROI_L = ndimage.binary_fill_holes(bg_ROI_L).astype(np.int32)
                bg_ROI_L[bg_ROI_L == 1] = -1
                bg_ROI_L[fg_ROI_L == 1] = 0  # rm fg contour pixels

            bg_ROI = bg_ROI_L[1:-1, 1:-1]  # 5,5 => 3,3
            fg_ROI = fg_ROI_L[1:-1, 1:-1]  # 5,5 => 3,3
            new_bg_pad_1[y_idx:(y_idx + ROI_sz), x_idx:(x_idx + ROI_sz)] = bg_ROI.astype(np.int32)
            if np.unique(fg_ROI)[0] == 1:  # rm total fg region
                new_fg_pad_1[y_idx + 1, x_idx + 1] = 0
                continue  # occ do not exist
            else:
                new_fg_pad_1[y_idx + 1, x_idx + 1] = 1

            bg_ROI_flat = np.delete(bg_ROI.flatten(), (ROI_sz * ROI_sz - 1) / 2)  # 8,

            # assign contour label to fg/bg pixs
            occ_order_pair_pad[y_idx:(y_idx + ROI_sz), x_idx:(x_idx + ROI_sz), 0].reshape(3, 3)[bg_ROI == -1] = 1

            diff = fg[y_idx, x_idx] - bg_ROI_flat  # diff between p and neighbor q; 8,
            order_label = np.zeros(ROI_sz * ROI_sz - 1)  # 8,
            order_label[diff == 2] = 1.  # center occludes neighbor
            if interpolate:  # use connect4 label to gen connect8 label
                order_label[diff == 2] = 1.
                order_label[0] = order_label[1] * order_label[3]
                order_label[2] = order_label[1] * order_label[4]
                order_label[5] = order_label[3] * order_label[6]
                order_label[7] = order_label[4] * order_label[6]

            occ_order_pair_pad[y_idx + 1, x_idx + 1, 1:] = order_label  # assign to fg pix

            # also assign occ order to neighborhood(8-connectivity) bg pix
            occ_order_pair_pad[y_idx, x_idx, 8]         = order_label[0] * bg_pad_1[y_idx, x_idx]
            occ_order_pair_pad[y_idx, x_idx + 1, 7]     = order_label[1] * bg_pad_1[y_idx, x_idx + 1]
            occ_order_pair_pad[y_idx, x_idx + 2, 6]     = order_label[2] * bg_pad_1[y_idx, x_idx + 2]
            occ_order_pair_pad[y_idx + 1, x_idx, 5]     = order_label[3] * bg_pad_1[y_idx + 1, x_idx]
            occ_order_pair_pad[y_idx + 1, x_idx + 2, 4] = order_label[4] * bg_pad_1[y_idx + 1, x_idx + 2]
            occ_order_pair_pad[y_idx + 2, x_idx, 3]     = order_label[5] * bg_pad_1[y_idx + 2, x_idx]
            occ_order_pair_pad[y_idx + 2, x_idx + 1, 2] = order_label[6] * bg_pad_1[y_idx + 2, x_idx + 1]
            occ_order_pair_pad[y_idx + 2, x_idx + 2, 1] = order_label[7] * bg_pad_1[y_idx + 2, x_idx + 2]

    occ_order_pair = occ_order_pair_pad[1:-1, 1:-1, :].astype(np.int32)
    new_bg = new_bg_pad_1[1:-1, 1:-1].astype(np.int32)
    new_fg = new_fg_pad_1[1:-1, 1:-1].astype(np.int32)

    return occ_order_pair, new_bg, new_fg


def occ_order_connect8_to_ori(occ_order_pix):
    """
    pixel pairwise occ order label to pixel-wise occ ori label, following occ ori convention
    used to gen occ ori dataset
    :param occ_order_pix: occ order prediction; H,W,8 ; [-1,0,1]; numpy
    :return: H,W ; [-1, 1]
    """
    occ_order_pix = occ_order_pix.astype(np.float32)

    # cal occ normal vec
    occ_normal_w  = -occ_order_pix[:, :, 3] + occ_order_pix[:, :, 4]
    occ_normal_h  = -occ_order_pix[:, :, 1] + occ_order_pix[:, :, 6]
    occ_normal_w += (- occ_order_pix[:, :, 0] + occ_order_pix[:, :, 2]
                     - occ_order_pix[:, :, 5] + occ_order_pix[:, :, 7]) * 0.707
    occ_normal_h += (- occ_order_pix[:, :, 0] - occ_order_pix[:, :, 2]
                     + occ_order_pix[:, :, 5] + occ_order_pix[:, :, 7]) * 0.707

    occ_normal_w = np.clip(occ_normal_w, -1., 1.)
    occ_normal_h = np.clip(occ_normal_h, -1., 1.)

    # normal theta to tangent theta
    theta_normal  = np.arctan2(occ_normal_h, occ_normal_w)  # w.r.t [-PI, PI] in WoH frame
    theta_tangent = theta_normal - PI / 2  # follow left-foreground rule
    theta_tangent[theta_tangent <= -PI] = theta_tangent[theta_tangent <= -PI] + 2*PI  # occ ori belongs to [-PI,PI]
    theta_tangent[(occ_normal_w == 0) * (occ_normal_h == 0)] = 0.  # default ori for viz
    theta_tangent = theta_tangent / PI  # => [-1,1]

    return theta_tangent


def occ_order_connect4_to_ori_tensor(occ_order):
    """
    pairwise occ order to pixel-wise occ ori for checking, following occ ori convention(currently only connectivity-4)
    used to gen occ ori dataset
    :param occ_order: occ order gt; N,2,H,W ; tensor [0,1,2]
    :return: occ orientation [-PI,PI] normalized to [-1, 1]; N,1,H,W
    """
    occ_order_w_edgewise = occ_order[:, 0, :, :].unsqueeze(1)  # N,1,H,W
    occ_order_h_edgewise = occ_order[:, 1, :, :].unsqueeze(1)  # N,1,H,W
    pad_w = nn.ConstantPad2d((1, 0, 0, 0), 0.)  # left, right, up, down
    pad_h = nn.ConstantPad2d((0, 0, 1, 0), 0.)  # left, right, up, down
    pred_w_pad = pad_w((occ_order_w_edgewise.float() - 1.))  # [0, 1, 2] => [-1, 0, 1]
    pred_h_pad = pad_h((occ_order_h_edgewise.float() - 1.))

    # cal occ normal vec
    occ_normal_w = torch.clamp(pred_w_pad[:, :, :, :-1] + pred_w_pad[:, :, :, 1:], -1., 1.)
    occ_normal_h = torch.clamp(pred_h_pad[:, :, :-1, :] + pred_h_pad[:, :, 1:, :], -1., 1.)

    # normal theta to tangent theta
    theta_normal  = torch.atan2(occ_normal_h, occ_normal_w)  # w.r.t [0, 1] in WoH frame, equivalent to occ ori def.
    theta_tangent = theta_normal - PI / 2  # follow left-foreground rule
    theta_tangent[theta_tangent <= -PI] = theta_tangent[theta_tangent <= -PI] + 2*PI  # occ ori belongs to [-PI,PI]
    theta_tangent[(occ_normal_w == 0) * (occ_normal_h == 0)] = 0.  # default ori for viz
    theta_tangent = theta_tangent / PI

    return theta_tangent


def occ_order_to_edge(occ_order, connectivity=4):
    """
    convert pairwise occ order to Figure/Ground notion and occ edge
    :param occ_order: pix occ order with its neighbor pixs: H,W,9
    :param connectivity: 4 or 8, neighborhood type
    """
    H, W, _ = occ_order.shape
    occ_edge    = np.zeros((H, W))  # all possible occ edge
    occ_edge_fg = np.zeros((H, W))  # fg occ edge
    occ_edge_bg = np.zeros((H, W))  # bg occ edge

    occ_exist_bool = np.any((occ_order != 0), axis=2)  # H,W
    if connectivity == 4:
        occ_status_pix = occ_order[:, :, 2] + occ_order[:, :, 4] + \
                         occ_order[:, :, 5] + occ_order[:, :, 7]
    elif connectivity == 8:
        occ_status_pix = np.sum(occ_order[:, :, 1:], axis=2)

    occ_edge_fg[occ_status_pix > 0] = 1
    occ_edge_bg[occ_status_pix < 0] = 1
    occ_edge[occ_exist_bool] = 1

    # filter out isolate pixels with a gaussian filter
    from scipy import ndimage, misc
    occ_edge_fg[occ_edge_fg == 1] = 255  # [0,1] => [0,255]
    occ_edge_bg[occ_edge_bg == 1] = 255
    fg_filtered = ndimage.uniform_filter(occ_edge_fg, size=3)
    bg_filtered = ndimage.uniform_filter(occ_edge_bg, size=3)
    occ_edge_fg[fg_filtered <= 29] = 0  # isolate point average value:28.3333
    occ_edge_bg[bg_filtered <= 29] = 0

    return occ_edge_fg/255, occ_edge_bg/255, occ_edge


def label_flip(occ_order_pair, type='horizontal'):
    """
    horizontal/vertical flip of occlusion order label(currently only 4-connectivity)
    used to gen augmented occ orde dataset
    :param occ_order_pair: H,W,9 ; [-1, 0, 1]
    :param type:
    :return:
    """
    occ_order_pair = occ_order_pair + 1  # => [0,1,2]
    if type == 'horizontal':
        for idx in range(0, occ_order_pair.shape[-1]):
            occ_order_tmp = Image.fromarray(occ_order_pair[:, :, idx])
            occ_order_tmp = np.array(occ_order_tmp.transpose(Image.FLIP_LEFT_RIGHT))

            if idx in [4, 5]:  # correct wrong labels after flip
                occ_order_tmp[occ_order_tmp == 0] = 255  # swap class label [0, 2] horizontal GT
                occ_order_tmp[occ_order_tmp == 2] = 0
                occ_order_tmp[occ_order_tmp == 255] = 2

            occ_order_pair[:, :, idx] = occ_order_tmp

    occ_order_pair += -1

    return occ_order_pair


def label_rotate(annot, rotate):
    """
    anti-clockwise rotate the occ order annotation by rotate*90 degrees
    :param annot: (H, W, 9) ; [-1, 0, 1]
    :param rotate: value in [0, 1, 2, 3]
    :return:
    """

    rotate = int(rotate)
    if rotate == 0:
        return annot
    else:
        annot_rot = np.rot90(annot, rotate)
        orientation = annot_rot[:, :, 1:].copy()
        if rotate == 1:
            mapping = [2, 4, 7, 1, 6, 0, 3, 5]
        elif rotate == 2:
            mapping = [7, 6, 5, 4, 3, 2, 1, 0]
        else:
            mapping = [5, 3, 0, 6, 1, 7, 4, 2]
        annot_rot[:, :, 1:] = orientation[:, :, mapping]
        return annot_rot
# ==================================================================================================================== #

# ========================================== functions for geometry computation ====================================== #
def insect_line_plane_3d(I0, I1, P0, n, epsilon=1e-6):
    """
    intersection point of line and plane in 3D in same coord sys(derived from multi_points_pose)
    following notation in https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    :param I0: two points on the line, numpy arr
    :param I1: two points on the line, numpy arr
    :param P0: a point on the plane (plane coordinate).
    :param n: a normal vector defining the plane direction(does not need to be normalized).
    :return P: intersection point vector or None (when the intersection can't be found).
    """

    I = I1 - I0  # line direction vector
    dot = np.dot(I, n)

    if abs(dot) > epsilon:
        d = np.dot(n, (P0 - I0)) / dot
        P = I0 + I * d  # intersection point
    else:  # The line is parallel to plane, totally inside or outside
        P = None
    return P


def insect_line_plane_3d_batch(I0, I1, P0, n, epsilon=1e-6):
    """
    intersection points of lines and planes in 3D in same coord sys(derived from multi_points_pose)
    following notation in https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection
    :param I0: point0 on the line, numpy arr N,3
    :param I1: point1 on the line, numpy arr N,3
    :param P0: a point on the plane (plane coordinate). numpy arr N,3 or 3,
    :param n: a normal vector defining the plane direction(does not need to be normalized). numpy arr N,3
    :return P: intersection point vector or nan(when line parallel to plane), numpy N,3
    """
    if P0.ndim == 1:
        P0 = np.repeat(P0[np.newaxis, :], I1.shape[0], axis=0)  # 3, => N,3
    if n.ndim == 1:
        n = np.repeat(n[np.newaxis, :], I1.shape[0], axis=0)  # 3, => N,3

    I = I1 - I0  # N,3
    dots = np.sum(I * n, axis=1)  # N,
    dots_tmp = np.copy(dots)

    dots_tmp[abs(dots) < epsilon] = 1.  # line parallel to plane

    d = (np.sum(n * (P0 - I0), axis=1) / dots_tmp).reshape(-1, 1)  # N,1
    P = I0 + I * d  # intersection points, N,3

    P[abs(dots) < epsilon, :] = np.array([np.nan, np.nan, np.nan])

    return P


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def cos_between_vects_batch(v1, v2):
    """
    Returns the cosine angle between vectors 'v1' and 'v2'
    :param v1: N,C
    :param v2: N,C
    :return N,
    """
    v1_u = v1 / np.linalg.norm(v1, axis=1).reshape(-1, 1)  # N,C
    v2_u = v2 / np.linalg.norm(v2, axis=1).reshape(-1, 1)

    dots = np.sum(v1_u * v2_u, axis=1)  # N,

    return np.clip(dots, -1.0, 1.0)


def cos_between_vects(v1, v2):
    """
    Returns the cosine angle between vectors 'v1' and 'v2'
    :param v1: C,
    :param v2: C,
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)


def angle_between_vects(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'
    angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    angle_between((1, 0, 0), (1, 0, 0))
    0.0
    angle_between((1, 0, 0), (-1, 0, 0))
    3.141592653589793
    """
    return np.arccos(cos_between_vects(v1, v2))


def depth_point2plane(depth, fx, fy):
    """point2point depth to point2plane depth"""
    H, W = depth.shape
    depth_plane = depth.copy()

    # compute field-of-view angel (rad)
    fov_x = 2 * atan((W/2) / fx)
    fov_y = 2 * atan((H/2) / fy)

    for i, j in itertools.product(range(H), range(W)):
        alpha_x = (pi - fov_x) / 2
        gamma_x = alpha_x + fov_x * ((W - j) / W)

        alpha_y = (pi - fov_y) / 2
        gamma_y = alpha_y + fov_y * ((H - i) / H)

        depth_plane[i, j] = np.sqrt(depth[i, j] ** 2 / (1 + 1 / (tan(gamma_x) ** 2) + 1 / (tan(gamma_y) ** 2)))

    return depth_plane.astype(depth.dtype)
# ==================================================================================================================== #


# ================================ functions for label conversion in train/val ======================================= #
def occ_order_pred_to_edge_prob(occ_order, connectivity=4):
    """
    edge-wise occ order prediction to pixel-wise occ edge probability
    :param occ_order: occ order prediction; N,C,H,W ; tensor
    :return: N,1,H,W ; [0.~1.]
    """
    # softmax
    occ_order_E_prob_edgewise = F.softmax(occ_order[:, 0:3, :, :], dim=1)  # N,1,H,W
    occ_order_S_prob_edgewise = F.softmax(occ_order[:, 3:6, :, :], dim=1)  # N,1,H,W
    non_occ_prob_E_edgewise = occ_order_E_prob_edgewise[:, 1, :, :].unsqueeze(1)  # N,1,H,W
    non_occ_prob_S_edgewise = occ_order_S_prob_edgewise[:, 1, :, :].unsqueeze(1)

    if connectivity == 8:
        occ_order_SE_prob_edgewise = F.softmax(occ_order[:, 6:9, :, :], dim=1)  # N,1,H,W
        occ_order_NE_prob_edgewise = F.softmax(occ_order[:, 9:12, :, :], dim=1)  # N,1,H,W
        non_occ_prob_SE_edgewise = occ_order_SE_prob_edgewise[:, 1, :, :].unsqueeze(1)  # N,1,H,W
        non_occ_prob_NE_edgewise = occ_order_NE_prob_edgewise[:, 1, :, :].unsqueeze(1)

    # average
    if connectivity == 4:
        non_occ_prob = (non_occ_prob_E_edgewise + non_occ_prob_S_edgewise) / 2
        occ_prob = 1. - non_occ_prob
    elif connectivity == 8:
        # padded pred prob
        nonocc_pad_E  = nn.ConstantPad2d((1, 0, 0, 0), 1.)  # left, right, up, down ; no occ prob = 1.
        nonocc_pad_S  = nn.ConstantPad2d((0, 0, 1, 0), 1.)  # left, right, up, down
        nonocc_pad_SE = nn.ConstantPad2d((1, 0, 1, 0), 1.)  # left, right, up, down
        nonocc_pad_NE = nn.ConstantPad2d((1, 0, 0, 1), 1.)  # left, right, up, down

        nonocc_pred_E_pad  = nonocc_pad_E(non_occ_prob_E_edgewise)  # N,1,H,W => N,1,H,W+1
        nonocc_pred_S_pad  = nonocc_pad_S(non_occ_prob_S_edgewise)  # N,1,H,W => N,1,H+1,W
        nonocc_pred_SE_pad = nonocc_pad_SE(non_occ_prob_SE_edgewise)  # N,1,H,W => N,1,H+1,W+1
        nonocc_pred_NE_pad = nonocc_pad_NE(non_occ_prob_NE_edgewise)  # N,1,H,W => N,1,H+1,W+1

        # average by 8 in connectivity-8
        occ_prob_E  = 1 - (nonocc_pred_E_pad[:, :, :, :-1] + nonocc_pred_E_pad[:, :, :, 1:]) / 2  # N,1,H,W
        occ_prob_S  = 1 - (nonocc_pred_S_pad[:, :, :-1, :] + nonocc_pred_S_pad[:, :, 1:, :]) / 2  # N,1,H,W
        occ_prob_SE = 1 - (nonocc_pred_SE_pad[:, :, :-1, :-1] + nonocc_pred_SE_pad[:, :, 1:, 1:]) / 2  # N,1,H,W
        occ_prob_NE = 1 - (nonocc_pred_NE_pad[:, :, 1:, :-1] + nonocc_pred_NE_pad[:, :, :-1, 1:]) / 2  # N,1,H,W

        occ_prob = (occ_prob_E + occ_prob_S + occ_prob_SE + occ_prob_NE) / 4

    return occ_prob


def occ_order_pred_to_ori(occ_order, connectivity=4):
    """
    pairwise occ order prediction to pixel-wise occ ori by voting, following occ ori convention
    :param occ_order: occ order prediction; N,6,H,W ; tensor
    :return: N,1,H,W ; [-PI, PI]
    """

    _, occ_order_E_edgewise = occ_order[:, 0:3, :, :].topk(1, dim=1, largest=True, sorted=True)  # N,1,H,W
    _, occ_order_S_edgewise = occ_order[:, 3:6, :, :].topk(1, dim=1, largest=True, sorted=True)  # N,1,H,W
    if connectivity == 8:
        _, occ_order_SE_edgewise = occ_order[:, 6:9, :, :].topk(1, dim=1, largest=True, sorted=True)  # N,1,H,W
        _, occ_order_NE_edgewise = occ_order[:, 9:12, :, :].topk(1, dim=1, largest=True, sorted=True)  # N,1,H,W

    pad_E = nn.ConstantPad2d((1, 0, 0, 0), 0.)  # left, right, up, down, no occ label
    pad_S = nn.ConstantPad2d((0, 0, 1, 0), 0.)
    pred_E_pad = pad_E((occ_order_E_edgewise.float() - 1.))  # [0, 1, 2] => [-1, 0, 1]
    pred_S_pad = pad_S((occ_order_S_edgewise.float() - 1.))
    if connectivity == 8:
        pad_SE = nn.ConstantPad2d((1, 0, 1, 0), 0.)  # left, right, up, down
        pad_NE = nn.ConstantPad2d((1, 0, 0, 1), 0.)  # left, right, up, down
        pred_SE_pad = pad_SE((occ_order_SE_edgewise.float() - 1.))  # [0, 1, 2] => [-1, 0, 1]
        pred_NE_pad = pad_NE((occ_order_NE_edgewise.float() - 1.))

    occ_normal_w = pred_E_pad[:, :, :, :-1] + pred_E_pad[:, :, :, 1:]
    occ_normal_h = pred_S_pad[:, :, :-1, :] + pred_S_pad[:, :, 1:, :]
    if connectivity == 8:
        occ_normal_se = pred_SE_pad[:, :, :-1, :-1] + pred_SE_pad[:, :, 1:, 1:]
        occ_normal_ne = pred_NE_pad[:, :, 1:, :-1] + pred_NE_pad[:, :, :-1, 1:]
        occ_normal_w += (occ_normal_se * 0.707 + occ_normal_ne * 0.707)  # cal normal vec coord, length sqrt(2)/2
        occ_normal_h += (occ_normal_se * 0.707 - occ_normal_ne * 0.707)

    occ_normal_w = torch.clamp(occ_normal_w, -1., 1.)
    occ_normal_h = torch.clamp(occ_normal_h, -1., 1.)

    # normal theta to tangent theta
    theta_normal = torch.atan2(occ_normal_h, occ_normal_w)  # w.r.t [0, 1] in WoH frame, equivalent to occ ori def.
    theta_tangent = theta_normal - PI / 2  # follow left-foreground rule
    theta_tangent[theta_tangent <= -PI] = theta_tangent[theta_tangent <= -PI] + 2*PI  # occ ori belongs to [-PI,PI]
    theta_tangent[(occ_normal_w == 0) * (occ_normal_h == 0)] = 0.  # default ori for viz

    return theta_tangent


def order4_to_order_pixelwise(occ_edge_prob, occ_order_E, occ_order_S):
    """
    convert connectivity-4 occ order prediction to pixelwise occ order label for downstream tasks
    :param occ_edge_prob: H,W ; [0~1] ; tensor
    :param occ_order_E: 1,H,W ; [0,1,2]
    :param occ_order_S: 1,H,W ; [0,1,2]
    :return: occ_order_pix; H,W,9 ; occ_prob
    """
    occ_edge_prob = occ_edge_prob.squeeze()
    H, W = occ_edge_prob.shape

    occ_edge_prob = occ_edge_prob * 2 - 1  # [0,1] => [-1,1]
    occ_order_E = occ_order_E - 1
    occ_order_S = occ_order_S - 1

    occ_order_E = occ_order_E.squeeze()  # => [-1,1]
    occ_order_S = occ_order_S.squeeze()
    occ_order_pix = torch.zeros((H, W, 9))

    occ_order_pix[:, :, 0] = occ_edge_prob[:, :] * 127  # [-1~1] => [-127~127]

    # N,S direction
    occ_order_pix[1:, :, 2] = -occ_order_S[:-1, :]
    occ_order_pix[:, :, 7] = occ_order_S[:, :]

    # W,E direction
    occ_order_pix[:, 1:, 4] = -occ_order_E[:, :-1]
    occ_order_pix[:, :, 5] = occ_order_E[:, :]

    # other directions
    occ_order_pix[:, :, 1] = torch.clamp(occ_order_pix[:, :, 2] + occ_order_pix[:, :, 4], -1, 1)
    occ_order_pix[:, :, 3] = torch.clamp(occ_order_pix[:, :, 2] + occ_order_pix[:, :, 5], -1, 1)
    occ_order_pix[:, :, 6] = torch.clamp(occ_order_pix[:, :, 4] + occ_order_pix[:, :, 7], -1, 1)
    occ_order_pix[:, :, 8] = torch.clamp(occ_order_pix[:, :, 5] + occ_order_pix[:, :, 7], -1, 1)

    occ_order_pix = np.array(occ_order_pix.detach().cpu()).astype(np.int8)

    return occ_order_pix


def order8_to_order_pixelwise(occ_edge_prob, occ_order_E, occ_order_S, occ_order_SE, occ_order_NE):
    """
    convert connectivity-8 pairwise occ order prediction to pixel-wise occ order label H,W,9 for downstream tasks
    :param occ_edge_prob: H,W ; [0~1] ; tensor
    :param occ_order_E: 1,H,W ; [0,1,2]
    :param occ_order_S: 1,H,W ; [0,1,2]
    :param occ_order_SE: 1,H,W ; [0,1,2]
    :param occ_order_NE: 1,H,W ; [0,1,2]
    :return: occ_order_pix; H,W,9 ; occ_edge_prob + occ_order along 8 neighbor directions ('occlude':1,'occluded':-1,'no occ':0)
    """
    occ_edge_prob = occ_edge_prob.squeeze()
    H, W = occ_edge_prob.shape

    occ_edge_prob = (occ_edge_prob * 2 - 1).squeeze()  # [0,1] => [-1,1]
    occ_order_E = (occ_order_E - 1).squeeze()  # 1,H,W => H,W
    occ_order_S = (occ_order_S - 1).squeeze()
    occ_order_SE = (occ_order_SE - 1).squeeze()
    occ_order_NE = (occ_order_NE - 1).squeeze()

    occ_order_pix = torch.zeros((H, W, 9))
    occ_order_pix[:, :, 0] = occ_edge_prob[:, :] * 127

    # N,S direction
    occ_order_pix[1:, :, 2] = -occ_order_S[:-1, :]
    occ_order_pix[:, :, 7] = occ_order_S[:, :]

    # W,E direction
    occ_order_pix[:, 1:, 4] = -occ_order_E[:, :-1]
    occ_order_pix[:, :, 5] = occ_order_E[:, :]

    # NW,SE direction
    occ_order_pix[1:, 1:, 1] = -occ_order_SE[:-1, :-1]
    occ_order_pix[:, :, 8] = occ_order_SE[:, :]

    # SW,NE direction
    occ_order_pix[:-1, 1:, 6] = -occ_order_NE[1:, :-1]
    occ_order_pix[:, :, 3] = occ_order_NE[:, :]

    occ_order_pix = np.array(occ_order_pix.detach().cpu()).astype(np.int8)

    return occ_order_pix
# ==================================================================================================================== #


# ========================================= functions for data visualization ========================================= #
def viz_occ_order(rgb, occ_order_pix, out_path):
    """
    viz occlusion order along each direction in pixel connectivity-8 neighborhood
    :param rgb: numpy, RGB,
    :param occ_order_pix: numpy, H,W,9
    :param out_path:
    :param fig_sz; out img sz
    :return:
    """
    from matplotlib import pyplot as plt
    H, W, C = rgb.shape
    fig_sz = np.array([W/100., H/100.]) * 3
    occ_order_pix = (occ_order_pix + 1.0) / 2.0  # [-1,1] => [0,1]

    fig = plt.figure(figsize=np.array(fig_sz), dpi=100, frameon=False)
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.imshow(occ_order_pix[:, :, 1], cmap='gray', vmin=0, vmax=1)
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.imshow(occ_order_pix[:, :, 2], cmap='gray', vmin=0, vmax=1)
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.imshow(occ_order_pix[:, :, 3], cmap='gray', vmin=0, vmax=1)
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.imshow(occ_order_pix[:, :, 4], cmap='gray', vmin=0, vmax=1)
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.imshow(rgb)  # BGR => RGB
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.imshow(occ_order_pix[:, :, 5], cmap='gray', vmin=0, vmax=1)
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.imshow(occ_order_pix[:, :, 6], cmap='gray', vmin=0, vmax=1)
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(occ_order_pix[:, :, 7], cmap='gray', vmin=0, vmax=1)
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.imshow(occ_order_pix[:, :, 8], cmap='gray', vmin=0, vmax=1)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.savefig(out_path.replace('rgb.png', 'order-check.png'))

    plt.close('all')


def viz_delta_depth(rgb, delta_depth, out_path, thresh):
    """
    viz depth difference along each direction in pixel connectivity-8 neighborhood
    :param rgb: numpy, RGB,
    :param delta_depth: numpy, H,W,9
    :param out_path:
    :param thresh: abs depth diff thresh(mm)
    :param fig_sz; out img sz
    :return:
    """
    from matplotlib import pyplot as plt
    H, W, C = rgb.shape
    fig_sz = np.array([W / 100., H / 100.]) * 3

    delta_depth = np.clip(delta_depth, -thresh, thresh)
    delta_depth = (delta_depth + thresh) / thresh / 2

    fig = plt.figure(figsize=np.array(fig_sz), dpi=100, frameon=False)
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.imshow(delta_depth[:, :, 0], cmap='gray', vmin=0, vmax=1)
    ax2 = fig.add_subplot(3, 3, 2)
    ax2.imshow(delta_depth[:, :, 1], cmap='gray', vmin=0, vmax=1)
    ax3 = fig.add_subplot(3, 3, 3)
    ax3.imshow(delta_depth[:, :, 2], cmap='gray', vmin=0, vmax=1)
    ax4 = fig.add_subplot(3, 3, 4)
    ax4.imshow(delta_depth[:, :, 3], cmap='gray', vmin=0, vmax=1)
    ax5 = fig.add_subplot(3, 3, 5)
    ax5.imshow(rgb)  # BGR => RGB
    ax6 = fig.add_subplot(3, 3, 6)
    ax6.imshow(delta_depth[:, :, 4], cmap='gray', vmin=0, vmax=1)
    ax7 = fig.add_subplot(3, 3, 7)
    ax7.imshow(delta_depth[:, :, 5], cmap='gray', vmin=0, vmax=1)
    ax8 = fig.add_subplot(3, 3, 8)
    ax8.imshow(delta_depth[:, :, 6], cmap='gray', vmin=0, vmax=1)
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.imshow(delta_depth[:, :, 7], cmap='gray', vmin=0, vmax=1)
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    fig.savefig(out_path)

    plt.close('all')


def depth_to_edge(depth, ROI_sz, thresh):
    """convert depth to occlusion boundary with depth difference threshold"""
    depth_pad   = cv2.copyMakeBorder(depth, 1, 1, 1, 1, cv2.BORDER_REPLICATE)  # with replicate padding
    edges_depth = np.zeros(depth.shape[:2])

    for y_idx in range(0, depth.shape[0]):
        for x_idx in range(0, depth.shape[1]):
            ROI = depth_pad[y_idx:(y_idx + ROI_sz), x_idx:(x_idx + ROI_sz)]
            ROI = ROI.astype(float)  # depth.dtype = uint16
            ROI_diff = np.abs(ROI - ROI[int((ROI_sz - 1) / 2), int((ROI_sz - 1) / 2)])
            if ROI_diff.max() > thresh:
                edges_depth[y_idx, x_idx] = 1.0  # boundary point

    return edges_depth
# ==================================================================================================================== #
