import torch
import os
import numpy as np
import sys
import argparse
import h5py
from scipy.io import loadmat
from skimage import feature
from scipy import ndimage
from tqdm import tqdm
import cv2

eigen_crop = [21, 461, 25, 617]


def init_device(cuda_device):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on " + torch.cuda.get_device_name(current_device))
    else:
        print("Running on CPU")
    return device


def load_datasets(nyu_gt_path, nyu_splits_path, depth_refine_path):
    nyu = h5py.File(nyu_gt_path, 'r')
    nyu_splits = loadmat(nyu_splits_path)
    depths = nyu['depths']
    index_dict = dict()
    nyu_test_gt = []
    count = 0
    eigen_crop = [21, 461, 25, 617]
    for i in tqdm(nyu_splits['testNdxs'], desc='loading GT depth maps'):
        index = i[0]-1
        index_dict[i[0]] = count
        count += 1
        label = depths[index]
        label = label.transpose(1, 0)
        label = label[eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
        nyu_test_gt.append(label)
    nyu_test_gt = np.array(nyu_test_gt)

    out_preds = dict()
    for method in tqdm(['eigen', 'laina', 'dorn', 'sharpnet', 'jiao', 'vnl'], desc='loading pred depth maps'):
        # load our depth maps saved in npy files
        pred_dir = os.path.join(depth_refine_path, method)
        pred = []
        for file_name in sorted([name for name in os.listdir(pred_dir) if name.endswith('.npy')]):
            pred.append(np.load(os.path.join(pred_dir, file_name)))
        pred = np.array(pred)

        # crop the depth map
        pred = pred[:, eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]

        out_preds['{}_pred'.format(method)] = pred

    return nyu_test_gt, out_preds, index_dict


def compute_errors(gt, pred):
   thresh = np.maximum((gt / (pred+1e-4)), (pred / (1e-4+gt)))
   a1 = (thresh < 1.25   ).mean()
   a2 = (thresh < 1.25 ** 2).mean()
   a3 = (thresh < 1.25 ** 3).mean()
   rmse = (gt - pred) ** 2
   rmse = np.sqrt(rmse.mean())
   rmse_log = (np.log(np.clip(gt+1e-4, a_min=1e-12, a_max=1e12)) - np.log(np.clip(pred+1e-4, a_min=1e-12, a_max=1e12))) ** 2
   rmse_log = np.sqrt(rmse_log.mean())
   abs_rel = np.mean(np.abs(gt - pred) / gt)
   log_10 = np.mean(np.abs(np.log10(np.clip(gt+1e-4, a_min=1e-12, a_max=1e12)) - np.log10(np.clip(pred+1e-4, a_min=1e-12, a_max=1e12))))
   return abs_rel, log_10, rmse, rmse_log, a1, a2, a3


def load_boundaries(index_dict, boundaries_list, boundaries_path):
    nyu_boundaries = open(boundaries_list, 'r').readlines()
    boundaries_imgs = []
    for i in nyu_boundaries:
        if 'oc' in i:
            boundaries_imgs.append(i.strip())
    boundaries_imgs.sort()
    list_index = []
    for i in nyu_boundaries:
        num = i.find('_')
        num_ = i.find('/')
        index = int(i[num_+1:num])
        list_index.append(index)
    list_index = list(set(list_index))
    list_index.sort()
    boundaries_gt = {}
    for i in boundaries_imgs:
        r = cv2.imread(boundaries_path + i, 0)
        r = r[eigen_crop[0]:eigen_crop[1], eigen_crop[2]:eigen_crop[3]]
        num = i.find('_')
        num_ = i.find('/')
        index = int(i[num_+1:num])
        boundaries_gt[index] = r
    boundaries_gt_ = []
    keys = list(boundaries_gt.keys())
    keys.sort()
    for i in keys:
        boundaries_gt_.append(boundaries_gt[i])
    boundaries_gt_ = np.array(boundaries_gt_)
    out_index = []
    for i in list_index:
        index = index_dict[i+1]
        out_index.append(index)
    out_index = np.array(out_index)
    return out_index, boundaries_gt_


def compute_depth_boundary_error(edges_gt, pred, low_thresh=0.15, high_thresh=0.3):
   # skip dbe if there is no ground truth distinct edge
    if np.sum(edges_gt) == 0:
        dbe_acc = np.nan
        dbe_com = np.nan
        edges_est = np.empty(pred.shape).astype(int)
    else:
        # normalize est depth map from 0 to 1
        pred_normalized = pred.copy().astype('f')
        pred_normalized[pred_normalized == 0] = np.nan
        pred_normalized = pred_normalized - np.nanmin(pred_normalized)
        pred_normalized = pred_normalized / np.nanmax(pred_normalized)
         # apply canny filter
        edges_est = feature.canny(pred_normalized, sigma=np.sqrt(2), low_threshold=low_thresh,
                                  high_threshold=high_thresh)

        # compute distance transform for chamfer metric
        D_gt = ndimage.distance_transform_edt(1 - edges_gt)
        D_est = ndimage.distance_transform_edt(1 - edges_est)

        max_dist_thr = 10.  # Threshold for local neighborhood
        mask_D_gt = D_gt < max_dist_thr  # truncate distance transform map

        E_fin_est_filt = edges_est * mask_D_gt  # compute shortest distance for all predicted edges

        if np.sum(E_fin_est_filt) == 0:  # assign MAX value if no edges could be detected in prediction
            dbe_acc = max_dist_thr
            dbe_com = max_dist_thr
        else:
            # accuracy: directed chamfer distance of predicted edges towards gt edges
            dbe_acc = np.nansum(D_gt * E_fin_est_filt) / np.nansum(E_fin_est_filt)

            # completeness: sum of undirected chamfer distances of predicted and gt edges
            ch1 = D_gt * edges_est  # dist(predicted,gt)
            ch1[ch1 > max_dist_thr] = max_dist_thr  # truncate distances
            ch2 = D_est * edges_gt  # dist(gt, predicted)
            ch2[ch2 > max_dist_thr] = max_dist_thr  # truncate distances
            res = ch1 + ch2  # summed distances
            dbe_com = np.nansum(res) / (np.nansum(edges_est) + np.nansum(edges_gt))  # normalized

    return dbe_acc, dbe_com


def eval_depth_and_boundaries(out_preds, out_index, nyu_test_gt, boundaries_gt_, eval_log):
    for key in tqdm(out_preds.keys(), desc='evaluating results'):
        ours = out_preds[key]
        pred_refine = []
        score = np.zeros(7)
        score_edge = np.zeros(2)
        for index in range(len(ours)):
            our_depth = ours[index]
            target_size = [440, 592]
            input_size = [480, 640]
            our_depth = cv2.resize(our_depth, (input_size[1], input_size[0]))
            out_depth = our_depth
            out_depth = cv2.resize(out_depth, (target_size[1], target_size[0]))
            pred_refine.append(out_depth)
            result = compute_errors(out_depth, nyu_test_gt[index])
            score += np.array(result)
        pred_refine = np.array(pred_refine)
        out_index = np.array(out_index, dtype=np.int32)
        pred_edge_input = pred_refine[out_index]
        for i in range(len(pred_edge_input)):
            result = compute_depth_boundary_error(boundaries_gt_[i]/255, pred_edge_input[i])
            score_edge += np.array(result)
        saved_stdout = sys.stdout
        sys.stdout = eval_log
        print(' ')
        print('Eval results...')
        print('Prediction: ' + key)
        print('=========================')
        for i in score:
            print(i / len(ours))
        for i in score_edge:
            print(i / len(pred_edge_input))
        print('*************************')
        print(' ')
        sys.stdout = saved_stdout
        eval_log.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', default='result/nyu', type=str)

    parser.add_argument('--nyu_gt_path', default='../data/NYUv2_OR/nyu_depth_v2_labeled.mat', type=str)
    parser.add_argument('--nyu_splits_path', default='../data/NYUv2_OR/nyuv2_splits.mat', type=str)

    parser.add_argument('--boundaries_path', default='../data/NYUv2_OR/NYUv2_OCpp/', type=str)
    parser.add_argument('--boundaries_list', default='../data/NYUv2_OR/NYUv2_OCpp/boundaries_list.txt', type=str)
    args = parser.parse_args()

    nyu_test_gt, out_preds, index_dict = load_datasets(args.nyu_gt_path, args.nyu_splits_path, args.input)
    out_index, boundaries_gt = load_boundaries(index_dict, args.boundaries_list, args.boundaries_path)
    print('Data loaded')
    print('Results will be saved at {}'.format(os.path.realpath(args.input)))
    eval_log_path = os.path.join(os.path.realpath(args.input), 'eval.txt')
    eval_log = open(eval_log_path, 'a')
    eval_depth_and_boundaries(out_preds, out_index, nyu_test_gt, boundaries_gt, eval_log)
