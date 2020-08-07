import numpy as np
import torch
import pickle as pkl
import h5py
from scipy.io import loadmat
import cv2
import gc


def neighbor_depth_variation(depth, diagonal=np.sqrt(2)):
    """Compute the variation of depth values in the neighborhood-8 of each pixel"""
    depth_crop = depth[1:-1, 1:-1, :]
    var1 = (depth_crop - depth[:-2, :-2, :]) / diagonal
    var2 = depth_crop - depth[:-2, 1:-1, :]
    var3 = (depth_crop - depth[:-2, 2:, :]) / diagonal
    var4 = depth_crop - depth[1:-1, :-2, :]
    var6 = depth_crop - depth[1:-1, 2:, :]
    var7 = (depth_crop - depth[2:, :-2, :]) / diagonal
    var8 = depth_crop - depth[2:, 1:-1, :]
    var9 = (depth_crop - depth[2:, 2:, :]) / diagonal

    return np.concatenate((var1, var2, var3, var4, var6, var7, var8, var9), -1)


def compute_tangent_adjusted_depth(depth_p, normal_p, depth_q, normal_q, eps=1e-3):
    # compute the depth map for the middl point
    depth_m = (depth_p + depth_q) / 2

    # compute the tangent-adjusted depth map for p and q
    ratio_p = np.linalg.norm(depth_p * normal_p, axis=-1, keepdims=True) / (
                np.linalg.norm(depth_m * normal_p, axis=-1, keepdims=True) + eps)
    depth_p_tangent = np.linalg.norm(depth_m * ratio_p, axis=-1, keepdims=True)

    ratio_q = np.linalg.norm(depth_q * normal_q, axis=-1, keepdims=True) / (
                np.linalg.norm(depth_m * normal_q, axis=-1, keepdims=True) + eps)
    depth_q_tangent = np.linalg.norm(depth_m * ratio_q, axis=-1, keepdims=True)

    return depth_p_tangent - depth_q_tangent


def neighbor_depth_variation_tangent(depth, normal, diagonal=np.sqrt(2)):
    """Compute the variation of tangent-adjusted depth values in the neighborhood-8 of each pixel"""
    depth_crop = depth[1:-1, 1:-1, :]
    normal_crop = normal[1:-1, 1:-1, :]
    var1 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[:-2, :-2, :], normal[:-2, :-2, :]) / diagonal
    var2 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[:-2, 1:-1, :], normal[:-2, 1:-1, :])
    var3 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[:-2, 2:, :], normal[:-2, 2:, :]) / diagonal
    var4 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[1:-1, :-2, :], normal[1:-1, :-2, :])
    var6 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[1:-1, 2:, :], normal[1:-1, 2:, :])
    var7 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[2:, :-2, :], normal[2:, :-2, :]) / diagonal
    var8 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[2:, 1:-1, :], normal[2:, 1:-1, :])
    var9 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[2:, 2:, :], normal[2:, 2:, :]) / diagonal

    return np.concatenate((var1, var2, var3, var4, var6, var7, var8, var9), -1)


def normalize_depth_map(depth):
    pred_normalized = depth.copy().astype('f')
    pred_normalized[pred_normalized == 0] = np.nan
    pred_normalized = pred_normalized - np.nanmin(pred_normalized)
    pred_normalized = pred_normalized / np.nanmax(pred_normalized)
    return pred_normalized


def padding_array(label_crop, h=480, w=640):
    eigen_crop = [21, 461, 25, 617]
    label = np.zeros((h, w, label_crop.shape[-1]))
    label[eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]] = label_crop
    label = torch.from_numpy(np.ascontiguousarray(label)).float().permute(2, 0, 1)
    return label


# functions to read depth predictions on NYUv2
eigen_crop = [0, 480, 0, 640]


def read_jiao():
    ours = []
    jiao_pred_path = '/space_sdd/NYU/depth_predictions/jiao_pred_mat/'
    for i in range(654):
        f = loadmat(jiao_pred_path + str(i+1) + '.mat')
        f = f['pred']
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
        line = cv2.resize(line, (640,480))
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
