# -*- coding: utf-8 -*-
"""
Created on Thu Nov 01 19:18:59 2018

@author: Tobias Koch, tobias.koch@tum.de
Remote Sensing Technology, Technical University of Munich
www.lmf.bgu.tum.de
"""

import numpy as np
from skimage import feature
from scipy import ndimage
from sklearn.decomposition import PCA
import math
import matplotlib.pyplot as plt
from scipy import ndimage


def compute_distance_related_errors(gt, pred):
    #initialize output
    abs_rel_vec_tmp = np.zeros(20, np.float32)
    log10_vec_tmp   = np.zeros(20, np.float32)
    rms_vec_tmp     = np.zeros(20, np.float32)
    
   
    # exclude masked invalid and missing measurements
    gt = gt[gt != 0]
    pred = pred[pred != 0]

    gt_all = gt
    pred_all = pred
    bot = 0.0
    idx = 0

    for top in range(1, 21):
        mask = np.logical_and(gt_all >= bot, gt_all <= top)
        gt_tmp = gt_all[mask]
        pred_tmp = pred_all[mask]
        # calc errors
        abs_rel_vec_tmp[idx], tmp, rms_vec_tmp[idx], log10_vec_tmp[idx], tmp, tmp, tmp = compute_global_errors(
            gt_tmp, pred_tmp)

        bot = top  # re-assign bottom threshold
        idx = idx + 1
     
    return abs_rel_vec_tmp,log10_vec_tmp,rms_vec_tmp
        

def compute_global_errors(gt, pred):
    # exclude masked invalid and missing measurements
    gt = gt[gt != 0]
    pred = pred[pred != 0]
    
    # compute global relative errors
    thresh = np.maximum((gt / pred), (pred / gt))
    thr1 = (thresh < 1.25).mean()
    thr2 = (thresh < 1.25 ** 2).mean()
    thr3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    # rmse_log = (np.log(gt) - np.log(pred)) ** 2
    # rmse_log = np.sqrt(rmse_log.mean())

    log10 = np.mean(np.abs(np.log10(np.clip(gt+1e-4, a_min=1e-12, a_max=1e12)) - np.log10(np.clip(pred+1e-4, a_min=1e-12, a_max=1e12))))

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred)**2) / gt)

    return abs_rel, sq_rel, rmse, log10, thr1, thr2, thr3


def compute_directed_depth_error(gt, pred, thr): 
    # exclude masked invalid and missing measurements
    gt = gt[gt != 0]
    pred = pred[pred != 0]
    
    # number of valid depth values 
    nPx = float(len(gt))

    gt[gt <= thr] = 1  # assign depths closer as 'thr' as '1s'
    gt[gt > thr] = 0  # assign depths farer as 'thr' as '0s'
    pred[pred <= thr] = 1
    pred[pred > thr] = 0
    
    diff = pred - gt  # compute difference map

    dde_0 = np.sum(diff == 0) / nPx
    dde_m = np.sum(diff == 1) / nPx
    dde_p = np.sum(diff == -1) / nPx
    
    return dde_0, dde_m, dde_p


def compute_depth_boundary_error(edges_gt, pred):
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
        edges_est = feature.canny(pred_normalized, sigma=np.sqrt(2), low_threshold=0.15, high_threshold=0.3)

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

    return dbe_acc, dbe_com, edges_est


def compute_planarity_error(gt,pred,paras,mask,calib):
    
    # mask invalid and missing depth values
    pred[pred==0]=np.nan
    gt[gt==0]=np.nan
    
    # number of planes of the current plane type
    nr_planes = paras.shape[0]
    
    # initialize PE errors
    pe_fla = np.empty(0)
    pe_ori = np.empty(0)
    
    for j in range(nr_planes):  #loop over number of planes
        
        # only consider depth values for this specific planar mask
        curr_plane_mask = mask.copy()
        curr_plane_mask[curr_plane_mask<(j+1)] = 0
        curr_plane_mask[curr_plane_mask>(j+1)] = 0
        remain_mask = curr_plane_mask.astype(float)
        remain_mask[remain_mask==0]=np.nan
        remain_mask[np.isnan(remain_mask)==0]=1    
    
        # only consider plane masks which are bigger than 5% of the image dimension
        if np.nansum(remain_mask)/(640.*480.)<0.05:
            flat = np.nan
            orie = np.nan
        else:
           #scale remaining depth map of current plane towards gt depth map
           mean_depth_est = np.nanmedian(pred*remain_mask)
           mean_depth_gt  = np.nanmedian(gt*remain_mask)
           est_depth_scaled = pred/(mean_depth_est/mean_depth_gt)*remain_mask
           
           # project masked and scaled depth values to 3D points
           fx_d = calib[0,0]
           fy_d = calib[1,1]
           cx_d = calib[2,0]
           cy_d = calib[2,1]
           #c,r = np.meshgrid(range(gt.shape[1]),range(gt.shape[0]))
           c,r = np.meshgrid(range(1,gt.shape[1]+1),range(1,gt.shape[0]+1))
           tmp_x = ((c-cx_d)*est_depth_scaled/fx_d) 
           tmp_y = est_depth_scaled
           tmp_z = (-(r-cy_d)*est_depth_scaled/fy_d)
           X = tmp_x.flatten()
           Y = tmp_y.flatten()
           Z = tmp_z.flatten()
           X = X[~np.isnan(X)] 
           Y = Y[~np.isnan(Y)] 
           Z = Z[~np.isnan(Z)] 
           pointCloud = np.stack((X, Y, Z))
           
           # fit 3D plane to 3D points (normal, d)
           pca = PCA(n_components=3)
           pca.fit(pointCloud.T)
           normal = -pca.components_[2,:] 
           point = np.mean(pointCloud,axis=1)
           d = -np.dot(normal,point);
           
           # PE_flat: deviation of fitted 3D plane
           flat = np.std(np.dot(pointCloud.T,normal.T)+d)*100.
           
           n_gt = paras[j,4:7]
           if np.dot(normal,n_gt)<0:
               normal = -normal
           
           # PE_ori: 3D angle error between ground truth plane and normal vector of fitted plane
           orie = math.atan2(np.linalg.norm(np.cross(n_gt,normal)),np.dot(n_gt,normal))*180./np.pi 
        
        pe_fla = np.append(pe_fla,flat) # append errors
        pe_ori = np.append(pe_ori,orie)
           
    return pe_fla,pe_ori