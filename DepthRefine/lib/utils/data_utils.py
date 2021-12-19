from os.path import join
import numpy as np
import torch
import pickle as pkl
import h5py
from scipy.io import loadmat
import cv2
import gc


# functions to read depth predictions on NYUv2
eigen_crop = [21, 461, 25, 617]


def read_jiao(nyu_depth_pred_dir):
    ours = []
    for i in range(654):
        f = loadmat(join(nyu_depth_pred_dir, 'jiao_pred_mat', '{}.mat'.format(i+1)))
        f = f['pred']
        f = f[eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
        ours.append(f)
    ours = np.array(ours)
    return ours


def read_laina(nyu_depth_pred_dir):
    laina_pred = h5py.File(join(nyu_depth_pred_dir, 'laina_predictions_NYUval.mat'), 'r')['predictions']
    laina_pred = np.array(laina_pred).transpose((0, 2, 1))
    laina_pred = laina_pred[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
    return laina_pred


def read_sharpnet(nyu_depth_pred_dir):
    with open(join(nyu_depth_pred_dir, 'sharpnet_prediction.pkl'), 'rb') as f:
        ours = pkl.load(f)
    ours = np.array(ours)
    ours = ours[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
    return ours


def read_eigen(nyu_depth_pred_dir):
    ours = loadmat(join(nyu_depth_pred_dir, 'eigen_nyud_depth_predictions.mat'))
    ours = ours['fine_predictions']
    ours = ours.transpose((2, 0, 1))
    out = []
    for line in ours:
        line = cv2.resize(line, (592, 440))
        out.append(line)
    out = np.array(out)
    return out


def read_dorn(nyu_depth_pred_dir):
    ours = []
    list_dirs = open(join(nyu_depth_pred_dir, 'NYUV2_DORN', 'list_dorn_order.txt'), 'r').readlines()
    for line in list_dirs:
        line = line.strip()
        f = loadmat(join(nyu_depth_pred_dir, 'NYUV2_DORN', 'NYUV2_DORN', line))
        pred = f['pred'][eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
        ours.append(pred)
    ours = np.array(ours)
    return ours


def read_bts(nyu_depth_pred_dir):
    ours = []
    list_dirs = open(join(nyu_depth_pred_dir, 'result_bts_nyu', 'pred_bts.txt'), 'r').readlines()
    tmp_dict = dict()
    for line in list_dirs:
        line = line.strip()
        num_tmp = line.rfind('_')
        key = int(line[num_tmp+1:-4])
        f = cv2.imread(join(nyu_depth_pred_dir, 'result_bts_nyu', 'raw', line), -1)
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


def read_vnl(nyu_depth_pred_dir):
    ours = pkl.load(open(join(nyu_depth_pred_dir, 'pred_VNL.pkl'), 'rb'))
    ours = np.array(ours) * 10
    ours = ours[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
    return ours


def padding_array(label_crop, h=480, w=640):
    eigen_crop = [21, 461, 25, 617]
    label = np.zeros((h, w, label_crop.shape[-1]))
    label[eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]] = label_crop
    label = torch.from_numpy(np.ascontiguousarray(label)).float().permute(2, 0, 1)
    return label