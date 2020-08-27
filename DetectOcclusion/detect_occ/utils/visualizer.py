# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------

import numpy as np
import os
import ntpath
import torch
import scipy.io as sio
import sys
from PIL import Image
import cv2
import matplotlib.pyplot as plt
plt.switch_backend('agg')  # use matplotlib without gui support
sys.path.append('../..')
from lib.dataset.gen_label_methods import order4_to_order_pixelwise, order8_to_order_pixelwise, occ_order_pred_to_edge_prob, occ_order_pred_to_ori

PI = 3.1416
curr_path = os.path.abspath(os.path.dirname(__file__))


def plot_train_metrics(train_writer, config, epoch, train_loss, train_mIoUs, train_mF1s, train_AP_edge):
    """plot loss and metrics on tensorboard"""
    train_writer.add_scalar('loss_epoch', train_loss, epoch)
    train_writer.add_scalar('AP_edge', train_AP_edge, epoch)

    if config.network.task_type == 'occ_order':
        train_order_mIoU = sum(train_mIoUs) / float(len(train_mIoUs))

        train_writer.add_scalar('meanIoU_all', train_order_mIoU, epoch)
        train_writer.add_scalar('meanIoU_E', train_mIoUs[0], epoch)
        train_writer.add_scalar('meanIoU_S', train_mIoUs[1], epoch)
        train_writer.add_scalar('meanF1_E', train_mF1s[0], epoch)
        train_writer.add_scalar('meanF1_S', train_mF1s[1], epoch)

        if config.dataset.connectivity == 8:
            train_writer.add_scalar('meanIoU_SE', train_mIoUs[2], epoch)
            train_writer.add_scalar('meanIoU_NE', train_mIoUs[3], epoch)
            train_writer.add_scalar('meanF1_SE', train_mF1s[2], epoch)
            train_writer.add_scalar('meanF1_NE', train_mF1s[3], epoch)

    elif config.network.task_type == 'occ_ori':
        train_writer.add_scalar('meanIoU_edge', train_mIoUs[0], epoch)
        train_writer.add_scalar('meanF1_edge', train_mF1s[0], epoch)


def plot_val_metrics(val_writer, config, epoch, val_loss, val_mIoUs, val_mF1s, val_AP_edge):
    val_writer.add_scalar('loss_epoch', val_loss, epoch)
    val_writer.add_scalar('AP_edge', val_AP_edge, epoch)

    if config.network.task_type == 'occ_order':
        val_order_mIoU = sum(val_mIoUs) / float(len(val_mIoUs))

        val_writer.add_scalar('loss_epoch', val_loss, epoch)
        val_writer.add_scalar('meanIoU_all', val_order_mIoU, epoch)
        val_writer.add_scalar('meanIoU_E', val_mIoUs[0], epoch)
        val_writer.add_scalar('meanIoU_S', val_mIoUs[1], epoch)
        val_writer.add_scalar('meanF1_E', val_mF1s[0], epoch)
        val_writer.add_scalar('meanF1_S', val_mF1s[1], epoch)

        if config.dataset.connectivity == 8:
            val_writer.add_scalar('meanIoU_SE', val_mIoUs[2], epoch)
            val_writer.add_scalar('meanIoU_NE', val_mIoUs[3], epoch)
            val_writer.add_scalar('meanF1_SE', val_mF1s[2], epoch)
            val_writer.add_scalar('meanF1_NE', val_mF1s[3], epoch)

    elif config.network.task_type == 'occ_ori':
        val_writer.add_scalar('meanIoU_edge', val_mIoUs[0], epoch)
        val_writer.add_scalar('meanF1_edge', val_mF1s[0], epoch)


def viz_and_log(inputs, net_out, targets, viz_writers, idx, epoch, config):
    """
    visualize train/val samples on tensorboard
    :param inputs: network input, tensor, N,C,H,W
    :param net_out: network input, tensor, N,C,H,W
    :param targets: tuple, (N,H,W ; N,H,W)
    """
    # viz model input
    mean_values = torch.tensor(config.dataset.pixel_means, dtype=inputs.dtype).view(3, 1, 1)
    std_values = torch.tensor(config.dataset.pixel_stds, dtype=inputs.dtype).view(3, 1, 1)
    model_in_vis = inputs[0, :3, :, :].cpu() * std_values + mean_values  # =>[0, 1]
    H, W = model_in_vis.shape[-2:]
    viz_writers[idx].add_image('Input', model_in_vis, epoch)

    if config.network.task_type == 'occ_order':
        edge_gt = targets[-1].unsqueeze(1).float()  # N,1,H,W

        # gen occ order pred
        order_gt_E = targets[0][0, :, :].view(-1, H, W).cpu()  # 1,H,W
        order_gt_S = targets[1][0, :, :].view(-1, H, W).cpu()
        _, ind_pred_h = net_out[0, :3, :, :].topk(1, dim=0, largest=True, sorted=True)  # 1,H,W
        _, ind_pred_v = net_out[0, 3:6, :, :].topk(1, dim=0, largest=True, sorted=True)
        if config.dataset.connectivity == 8:
            order_gt_SE = targets[2][0, :, :].view(-1, H, W).cpu()  # 1,H,W
            order_gt_NE = targets[3][0, :, :].view(-1, H, W).cpu()
            _, ind_pred_SE = net_out[0, 6:9, :, :].topk(1, dim=0, largest=True, sorted=True)  # 1,H,W
            _, ind_pred_NE = net_out[0, 9:12, :, :].topk(1, dim=0, largest=True, sorted=True)

        # gen occ edge prob from occ order
        edge_prob_pred = occ_order_pred_to_edge_prob(net_out, config.dataset.connectivity)  # N,1,H,W

        # plot
        viz_writers[idx].add_image('occ_order_E.1.gt', order_gt_E.float() / 2, epoch)
        viz_writers[idx].add_image('occ_order_S.1.gt', order_gt_S.float() / 2, epoch)
        viz_writers[idx].add_image('occ_order_E.2.pred', ind_pred_h.float() / 2, epoch)
        viz_writers[idx].add_image('occ_order_S.2.pred', ind_pred_v.float() / 2, epoch)
        if config.dataset.connectivity == 8:
            viz_writers[idx].add_image('occ_order_SE.1.gt', order_gt_SE.float() / 2, epoch)
            viz_writers[idx].add_image('occ_order_NE.1.gt', order_gt_NE.float() / 2, epoch)
            viz_writers[idx].add_image('occ_order_SE.2.pred', ind_pred_SE.float() / 2, epoch)
            viz_writers[idx].add_image('occ_order_NE.2.pred', ind_pred_NE.float() / 2, epoch)

        viz_writers[idx].add_image('Occ_edge.1.gt', edge_gt[0], epoch)
        viz_writers[idx].add_image('Occ_edge.2.pred', edge_prob_pred[0], epoch)

    elif config.network.task_type == 'occ_ori':
        edge_gt        = targets[-1].unsqueeze(1).float()  # N,1,H,W
        edge_prob_pred = net_out[:, 0, :, :].unsqueeze(1)  # N,1,H,W
        ori_gt = targets[0].unsqueeze(1).float()  # N,1,H,W
        ori_gt = (torch.clamp(ori_gt, -PI, PI) + PI) / PI / 2  # [-PI,PI] => [0,1]
        ori_pred = net_out[:, 1, :, :].unsqueeze(1)  # N,1,H,W
        ori_pred = (torch.clamp(ori_pred, -PI, PI) + PI) / PI / 2  # [-PI,PI] => [0,1]

        viz_writers[idx].add_image('Occ_edge.1.gt', edge_gt[0], epoch)
        viz_writers[idx].add_image('Occ_edge.2.pred', edge_prob_pred[0], epoch)
        viz_writers[idx].add_image('occ_ori.1.gt', ori_gt[0], epoch)
        viz_writers[idx].add_image('occ_ori.2.pred', ori_pred[0], epoch)


def viz_and_save(net_in, net_out, img_abs_path, out_dir, config, epoch):
    """save current sample predictions in res dir w/ .mat predictions(if exist)"""
    file_name = img_abs_path[0].split('/')[-1]
    img_suffix = '.{}'.format(file_name.split('.')[-1])
    img_name = file_name.replace(img_suffix, '')

    # set res dirs
    valset_name = config.dataset.test_dataset
    root_eval_dir = os.path.join(out_dir, 'test_{}_{}'.format(epoch, valset_name))
    lbl_eval_dir = os.path.join(root_eval_dir, 'res_mat')
    img_eval_dir = os.path.join(root_eval_dir, 'images')
    if not os.path.exists(root_eval_dir): os.makedirs(root_eval_dir)
    if not os.path.exists(lbl_eval_dir): os.makedirs(lbl_eval_dir)
    if not os.path.exists(img_eval_dir): os.makedirs(img_eval_dir)

    # get original img
    mean_values = torch.tensor(config.dataset.pixel_means, dtype=net_in.dtype).view(3, 1, 1)
    std_values  = torch.tensor(config.dataset.pixel_stds, dtype=net_in.dtype).view(3, 1, 1)
    img_in_viz = net_in[0, :3, :, :].cpu() * std_values + mean_values
    img_in_viz = np.transpose(img_in_viz.numpy(), (1, 2, 0)).astype(np.float32)  # 3,H,W => H,W,3
    H, W = img_in_viz.shape[:2]

    if config.network.task_type == 'occ_order':
        # gen occ edge/ori/order pred
        occ_edge_prob_pred = occ_order_pred_to_edge_prob(net_out, config.dataset.connectivity)  # N,1,H,W
        occ_ori_pred       = occ_order_pred_to_ori(net_out, config.dataset.connectivity)  # N,1,H,W
        _, ind_pred_E = net_out[0, :3, :, :].topk(1, dim=0, largest=True, sorted=True)  # 1,H,W; [0,1,2]
        _, ind_pred_S = net_out[0, 3:6, :, :].topk(1, dim=0, largest=True, sorted=True)
        if config.dataset.connectivity == 8:
            _, ind_pred_SE = net_out[0, 6:9, :, :].topk(1, dim=0, largest=True, sorted=True)  # 1,H,W; [0,1,2]
            _, ind_pred_NE = net_out[0, 9:12, :, :].topk(1, dim=0, largest=True, sorted=True)

        if config.dataset.connectivity == 4:
            occ_order_pred = order4_to_order_pixelwise(occ_edge_prob_pred, ind_pred_E, ind_pred_S)  # H,W,9
        elif config.dataset.connectivity == 8:
            occ_order_pred = order8_to_order_pixelwise(occ_edge_prob_pred, ind_pred_E, ind_pred_S, ind_pred_SE, ind_pred_NE)  # H,W,9

        # viz occ edge/ori/order
        occ_edge_prob_pred_viz = np.array(occ_edge_prob_pred.cpu())[0, 0, :, :] * 255.  # [0,1] => [0,255]
        occ_ori_pred_viz       = (np.array(occ_ori_pred.cpu())[0, 0, :, :] / PI + 1) / 2. * 255.  # [-PI,PI] => [0,255]
        ind_pred_E = np.array(ind_pred_E.cpu()).reshape(H, W).astype(np.float32)  # 1,H,W => H,W,1
        ind_pred_S = np.array(ind_pred_S.cpu()).reshape(H, W).astype(np.float32)  # 1,H,W => H,W,1
        if config.dataset.connectivity == 8:
            ind_pred_SE = np.array(ind_pred_SE.cpu()).reshape(H, W).astype(np.float32)  # 1,H,W => H,W,1
            ind_pred_NE = np.array(ind_pred_NE.cpu()).reshape(H, W).astype(np.float32)  # 1,H,W => H,W,1

        # save pred occ order imgs
        occ_order_E4eval_path = os.path.join(img_eval_dir, '{}_lab_v_g_order_E.png'.format(img_name))
        occ_order_S4eval_path = os.path.join(img_eval_dir, '{}_lab_v_g_order_S.png'.format(img_name))
        Image.fromarray((ind_pred_E / 2. * 255.).astype(np.uint8), mode='L').save(occ_order_E4eval_path)
        Image.fromarray((ind_pred_S / 2. * 255.).astype(np.uint8), mode='L').save(occ_order_S4eval_path)
        if config.dataset.connectivity == 8:
            occ_order_SE4eval_path = os.path.join(img_eval_dir, '{}_lab_v_g_order_SE.png'.format(img_name))
            occ_order_NE4eval_path = os.path.join(img_eval_dir, '{}_lab_v_g_order_NE.png'.format(img_name))
            Image.fromarray((ind_pred_SE / 2. * 255.).astype(np.uint8), mode='L').save(occ_order_SE4eval_path)
            Image.fromarray((ind_pred_NE / 2. * 255.).astype(np.uint8), mode='L').save(occ_order_NE4eval_path)

        # save pred occ order npy before NMS
        occ_order_eval_dir = os.path.join(lbl_eval_dir, 'test_order_pred')
        if not os.path.exists(occ_order_eval_dir): os.makedirs(occ_order_eval_dir)
        occ_order_pred_path = os.path.join(occ_order_eval_dir, '{}-order-pix.npy'.format(img_name))
        np.save(occ_order_pred_path, occ_order_pred)

    elif config.network.task_type == 'occ_ori':
        occ_edge_prob_pred_viz = np.array(net_out.cpu())[0, 0, :, :] * 255.  # [0,1] => [0,255]
        occ_ori_pred           = torch.clamp(net_out[0, 1, :, :], -PI, PI)
        occ_ori_pred_viz       = (np.array(occ_ori_pred.cpu()) / PI + 1) / 2. * 255.  # [-PI,PI] => [0,255]

    # save img, pred occ edge/ori
    rgb_path           = os.path.join(img_eval_dir, '{}_img_v.png'.format(img_name))
    occ_edge_prob_path = os.path.join(img_eval_dir, '{}_lab_v_g.png'.format(img_name))
    occ_ori_path       = os.path.join(img_eval_dir, '{}_lab_v_g_ori.png'.format(img_name))
    Image.fromarray((img_in_viz * 255.).astype(np.uint8)).save(rgb_path)
    Image.fromarray(occ_edge_prob_pred_viz.astype(np.uint8), mode='L').save(occ_edge_prob_path)
    Image.fromarray(occ_ori_pred_viz.astype(np.uint8), mode='L').save(occ_ori_path)

    # save .mat res for edge/ori eval if matlab GT exists
    if config.dataset.val_w_mat_gt:
        matlab_tool = MATLAB(root_eval_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (valset_name, 'test', epoch))
        preds_list  = [occ_edge_prob_pred_viz, occ_ori_pred_viz]
        save_res_matlab(matlab_tool, {'lab_v_g': preds_list}, img_name)


def plot_conf_mat(confusion_mat, out_path, epoch, title=''):
    """plot confusion matrix and save"""
    fig_conf = plt.figure()
    axe = fig_conf.add_subplot(111)
    cax = axe.matshow(confusion_mat, vmin=0, vmax=1)
    fig_conf.colorbar(cax), plt.xlabel('Predicted class'), plt.ylabel('GT class')
    axe.set_title(title)
    fig_conf.savefig(os.path.join(out_path, '{}_{}.png'.format(title, epoch)))


def check_occ_order_pred(order_gt, order_pred):
    """
    convert gt and pred to colormap to check 3-class pairwise occ order pred
    :param order_gt: tensor H,W
    :param order_pred: tensor H,W
    :return: numpy 3,H,W
    """
    H, W = order_gt.shape
    order_gt = np.array(order_gt.cpu()).reshape(H, W)
    order_pred = np.array(order_pred.cpu())
    order_color = np.ones((H, W, 3)) / 2  # gray for no occ
    order_color[order_gt == 0, :] = np.array([0, 0, 0])  # black for class 0
    order_color[order_gt == 2, :] = np.array([1, 1, 1])  # white for class 2
    order_color[(order_gt == 0) * (order_pred == 0), :] = np.array([0, 1, 0])  # green for Correct pred
    order_color[(order_gt == 0) * (order_pred == 2), :] = np.array([1, 0, 0])  # red for inverse cls pred
    order_color[(order_gt == 1) * (order_pred != 1), :] = np.array([1, 0, 0])
    order_color[(order_gt == 2) * (order_pred == 2), :] = np.array([0, 1, 0])
    order_color[(order_gt == 2) * (order_pred == 0), :] = np.array([1, 0, 0])

    return order_color.astype(np.float32).transpose((2, 0, 1))  # H,W,3 => 3,H,W


def save_res_matlab(matlab_tool, labs_pred, image_path):
    """
    save curr sample pred and gt as .mat for eval in matlab
    :param matlab_tool:
    :param labs_pred: dict of preds {label: value} ; value [0~255]
    :param images_path:
    :return:
    """
    PI = 3.1416
    short_path = ntpath.basename(image_path)  # return file name
    f_name = os.path.splitext(short_path)[0]  # rm .ext
    if 'rgb' in f_name: f_name = f_name.replace('-rgb', '')
    org_gt_name = '{}{}'.format(f_name, matlab_tool.gt_type)
    org_gt_path = os.path.join(matlab_tool.org_gt_dir, org_gt_name)
    out_gt_path = os.path.join(matlab_tool.out_gt_dir, '{}.mat'.format(f_name))

    for out_label, preds_np in labs_pred.items():
        pred_name = '%s_%s.mat' % (f_name, out_label)
        out_pred_path = os.path.join(matlab_tool.out_pred_dir, pred_name)

        pred_edge = preds_np[0]  # [0,255]
        pred_ori  = preds_np[1]  # [0,255]
        H, W = pred_edge.shape[:2]

        if '.mat' in org_gt_name:
            org_gt = sio.loadmat(org_gt_path)
            gts = org_gt['gtStruct']['gt_theta'][0][0][0]
            out_gt = np.zeros((H, W, gts.shape[0] * 2))
            for anno_idx in range(0, gts.shape[0]):  # annotator idx
                edge_map = gts[anno_idx][:, :, 0]  # [0, 1]
                ori_map  = gts[anno_idx][:, :, 1]  # [-PI, PI]

                out_gt[:, :, anno_idx * 2]     = edge_map
                out_gt[:, :, anno_idx * 2 + 1] = ori_map
        elif '.png' in org_gt_name:  # .png gt
            edge_map = cv2.imread(org_gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255  # [0, 255] => [0, 1]
            if len(edge_map.shape) == 3: edge_map = edge_map[:, :, 0]
            ori_map = np.zeros((H, W))
            out_gt = np.zeros((H, W, 2))
            out_gt[:, :, 0] = edge_map
            out_gt[:, :, 1] = ori_map

        sio.savemat(out_gt_path, {'gt_edge_theta': out_gt})

        pred_edge = pred_edge / 255.0  # [0, 255] => [0, 1]
        pred_ori  = pred_ori.astype(np.float32) / 255. * 2 * PI - PI  # [0, 255] => [-PI, PI]
        sio.savemat(out_pred_path, {'edge_ori': {'edge': pred_edge, 'ori': pred_ori}})


class MATLAB:
    def __init__(self, res_dir, valset_name):
        self.title = valset_name
        self.res_dir = res_dir
        self.out_root_dir = os.path.join(self.res_dir, 'res_mat')
        self.out_gt_dir = os.path.join(self.out_root_dir, 'test_gt')
        self.out_pred_dir = os.path.join(self.out_root_dir, 'test_pred')
        self.out_order_dir = os.path.join(self.out_root_dir, 'test_order_pred')

        if 'nyu' in valset_name:
            self.org_gt_dir = os.path.join(curr_path, '../..', 'data/NYUv2_OR/data/')
            self.gt_type = '.mat'
        elif 'BSDSownership' in valset_name:
            self.org_gt_dir = os.path.join(curr_path, '../..', 'data/BSDS300/BSDS_theta/testOri_mat')
            self.gt_type = '.mat'
        elif 'ibims' in valset_name or 'interiornet' in valset_name:
            self.org_gt_dir = os.path.join(curr_path, '../..', 'data/iBims1_OR/data/')
            self.gt_type = '.mat'

        if not os.path.exists(self.out_root_dir): os.makedirs(self.out_root_dir)
        if not os.path.exists(self.out_gt_dir): os.makedirs(self.out_gt_dir)
        if not os.path.exists(self.out_pred_dir): os.makedirs(self.out_pred_dir)
        if not os.path.exists(self.out_order_dir): os.makedirs(self.out_order_dir)


