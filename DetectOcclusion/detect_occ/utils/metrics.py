# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------

import torch
import numpy as np
from sklearn.metrics import average_precision_score
import sys
sys.path.append('../..')
from .utility import AverageMeter
from lib.dataset.gen_label_methods import occ_order_pred_to_edge_prob


class MetricsEvaluator(object):
    """object for train/val evaluation with relevant metric"""

    def __init__(self, config, isTrain):
        self.config = config
        self.isTrain = isTrain

        # occ edge metrics
        self.mIoU_edge = AverageMeter()
        self.mF1_edge  = AverageMeter()
        self.AP_edge   = AverageMeter()

        self.perf_edge = PixLabelEvaluator(num_class=2, cls_sel_ind=[1])

        self.curr_mIoU_edge = 0.
        self.curr_mF1_edge  = 0.
        self.curr_AP_edge   = 0.

        # occ order metrics
        if self.config.network.task_type == 'occ_order':
            self.mIoU_E  = AverageMeter()
            self.mIoU_S  = AverageMeter()
            self.mIoU_SE = AverageMeter()
            self.mIoU_NE = AverageMeter()

            self.mF1_E  = AverageMeter()
            self.mF1_S  = AverageMeter()
            self.mF1_SE = AverageMeter()
            self.mF1_NE = AverageMeter()

            self.perf_E  = PixLabelEvaluator(3, config.TEST.class_sel_ind)
            self.perf_S  = PixLabelEvaluator(3, config.TEST.class_sel_ind)
            self.perf_SE = PixLabelEvaluator(3, config.TEST.class_sel_ind)
            self.perf_NE = PixLabelEvaluator(3, config.TEST.class_sel_ind)

            self.curr_mIoU_E  = 0.
            self.curr_mIoU_S  = 0.
            self.curr_mIoU_SE = 0.
            self.curr_mIoU_NE = 0.

            self.curr_mF1_E  = 0.
            self.curr_mF1_S  = 0.
            self.curr_mF1_SE = 0.
            self.curr_mF1_NE = 0.

    def cal_batch_metrics(self, net_out, targets):
        if self.config.network.task_type == 'occ_order':
            # cal occ order metrics
            self.perf_E.cal_batch(targets[0], net_out[:, 0:3, :, :])
            self.perf_S.cal_batch(targets[1], net_out[:, 3:6, :, :])

            self.curr_mIoU_E, _ = self.perf_E.Mean_IoU()
            self.curr_mIoU_S, _ = self.perf_S.Mean_IoU()
            self.curr_mF1_E, _  = self.perf_E.Mean_F1_Score()
            self.curr_mF1_S, _  = self.perf_S.Mean_F1_Score()

            self.mIoU_E.update(self.curr_mIoU_E.item(), targets[0].size(0))
            self.mIoU_S.update(self.curr_mIoU_S.item(), targets[1].size(0))
            self.mF1_E.update(self.curr_mF1_E.item(), targets[0].size(0))
            self.mF1_S.update(self.curr_mF1_S.item(), targets[1].size(0))

            if self.config.dataset.connectivity == 8:
                self.perf_SE.cal_batch(targets[2], net_out[:, 6:9, :, :])
                self.perf_NE.cal_batch(targets[3], net_out[:, 9:12, :, :])

                self.curr_mIoU_SE, _ = self.perf_SE.Mean_IoU()
                self.curr_mIoU_NE, _ = self.perf_NE.Mean_IoU()
                self.curr_mF1_SE, _  = self.perf_SE.Mean_F1_Score()
                self.curr_mF1_NE, _  = self.perf_NE.Mean_F1_Score()

                self.mIoU_SE.update(self.curr_mIoU_SE.item(), targets[2].size(0))
                self.mIoU_NE.update(self.curr_mIoU_NE.item(), targets[3].size(0))
                self.mF1_SE.update(self.curr_mF1_SE.item(), targets[2].size(0))
                self.mF1_NE.update(self.curr_mF1_NE.item(), targets[3].size(0))

            # cal occ edge metrics
            if not self.isTrain:
                occ_edge_prob      = occ_order_pred_to_edge_prob(net_out, self.config.dataset.connectivity)
                occ_edge_prob_flat = np.array(occ_edge_prob.detach().cpu()).flatten()
                occ_edge_gt_flat   = np.array(targets[-1].detach().cpu()).flatten().astype(np.int16)

                self.curr_AP_edge = average_precision_score(occ_edge_gt_flat, occ_edge_prob_flat) * 100  # percentage
                self.AP_edge.update(self.curr_AP_edge, targets[-1].size(0))

        elif self.config.network.task_type == 'occ_ori':
            # cal occ edge metrics
            occ_edge_prob = net_out[:, 0, :, :].unsqueeze(dim=1)  # N,1,H,W
            occ_edge_hard = (occ_edge_prob > 0.5).float()
            occ_edge_gt   = targets[-1]  # N,H,W

            self.perf_edge.cal_batch(occ_edge_gt, occ_edge_hard)

            self.curr_mIoU_edge, _ = self.perf_edge.Mean_IoU()
            self.curr_mF1_edge, _  = self.perf_edge.Mean_F1_Score()
            self.mIoU_edge.update(self.curr_mIoU_edge, targets[-1].size(0))
            self.mF1_edge.update(self.curr_mF1_edge, targets[-1].size(0))

            if not self.isTrain:
                occ_edge_gt_flat   = np.array(targets[-1].detach().cpu()).flatten().astype(np.int16)
                occ_edge_prob_flat = np.array(occ_edge_prob.detach().cpu()).flatten()
                self.curr_AP_edge  = average_precision_score(occ_edge_gt_flat, occ_edge_prob_flat) * 100  # percentage
                self.AP_edge.update(self.curr_AP_edge, targets[-1].size(0))

    def cal_set_metrics(self, isTest=False):
        """cal avg perf over whole train/val set"""
        if self.config.network.task_type == 'occ_order':
            if self.config.dataset.connectivity == 4:
                self.mIoU_all = (self.mIoU_E.avg + self.mIoU_S.avg) / 2
                self.mF1_all  = (self.mF1_E.avg + self.mF1_S.avg) / 2
                self.mIoUs    = [self.mIoU_E.avg, self.mIoU_S.avg]
                self.mF1s     = [self.mF1_E.avg, self.mF1_S.avg]

                if isTest:
                    self.perf_E.confusion_matrix_curr = self.perf_E.confusion_matrix_all
                    self.perf_S.confusion_matrix_curr = self.perf_S.confusion_matrix_all
                    _, prec_E_classes   = self.perf_E.Mean_Precision()
                    _, prec_S_classes   = self.perf_S.Mean_Precision()
                    _, recall_E_classes = self.perf_E.Mean_Recall()
                    _, recall_S_classes = self.perf_S.Mean_Recall()

                    self.prec_all   = (prec_E_classes + prec_S_classes) / 2
                    self.recall_all = (recall_E_classes + recall_S_classes) / 2

            elif self.config.dataset.connectivity == 8:
                self.mIoU_all = (self.mIoU_E.avg + self.mIoU_S.avg +
                                 self.mIoU_SE.avg + self.mIoU_NE.avg) / 4
                self.mF1_all  = (self.mF1_E.avg + self.mF1_S.avg +
                                 self.mF1_SE.avg + self.mF1_NE.avg) / 4
                self.mIoUs    = [self.mIoU_E.avg, self.mIoU_S.avg,
                                 self.mIoU_SE.avg, self.mIoU_NE.avg]
                self.mF1s     = [self.mF1_E.avg, self.mF1_S.avg,
                                 self.mF1_SE.avg, self.mF1_NE.avg]

                if isTest:
                    self.perf_E.confusion_matrix_curr  = self.perf_E.confusion_matrix_all
                    self.perf_S.confusion_matrix_curr  = self.perf_S.confusion_matrix_all
                    self.perf_SE.confusion_matrix_curr = self.perf_SE.confusion_matrix_all
                    self.perf_NE.confusion_matrix_curr = self.perf_NE.confusion_matrix_all
                    _, prec_E_classes    = self.perf_E.Mean_Precision()
                    _, prec_S_classes    = self.perf_S.Mean_Precision()
                    _, prec_SE_classes   = self.perf_SE.Mean_Precision()
                    _, prec_NE_classes   = self.perf_NE.Mean_Precision()
                    _, recall_E_classes  = self.perf_E.Mean_Recall()
                    _, recall_S_classes  = self.perf_S.Mean_Recall()
                    _, recall_SE_classes = self.perf_SE.Mean_Recall()
                    _, recall_NE_classes = self.perf_NE.Mean_Recall()

                    self.prec_all   = (prec_E_classes + prec_S_classes +
                                       prec_SE_classes + prec_NE_classes) / 4
                    self.recall_all = (recall_E_classes + recall_S_classes +
                                       recall_SE_classes + recall_NE_classes) / 4

        elif self.config.network.task_type == 'occ_ori':
            self.mIoU_all = self.mIoU_edge.avg
            self.mF1_all  = self.mF1_edge.avg
            self.mIoUs = [self.mIoU_edge.avg]
            self.mF1s  = [self.mF1_edge.avg]


class PixLabelEvaluator(object):
    """
    pixel-wise labeling evaluator
    derived from https://github.com/jfzhang95/pytorch-deeplab-xception/blob/master/utils/metrics.py
    """
    def __init__(self, num_class, cls_sel_ind):
        self.num_class = num_class
        self.cls_sel_ind = cls_sel_ind
        self.confusion_matrix_curr = np.zeros((self.num_class,) * 2)
        self.confusion_matrix_all = np.zeros((self.num_class,) * 2)

    def Pixel_Accuracy(self):
        """sum_cls(TP_cls) / sum_cls(TP_cls + FN_cls)"""
        Acc = np.diag(self.confusion_matrix_curr).sum() / self.confusion_matrix_curr.sum() * 100.0
        return Acc

    def Mean_Pixel_Accuracy(self):
        """sum_cls[TP_cls / (TP_cls + FN_cls)] / cls_num"""
        Accs = np.diag(self.confusion_matrix_curr) / self.confusion_matrix_curr.sum(axis=1)
        Accs_sel = Accs[self.cls_sel_ind]
        MAcc = np.nanmean(Accs_sel) * 100.0
        if np.isnan(MAcc): MAcc = np.array(0.0)  # in case of no valid class for curr sample
        return MAcc

    def Mean_Precision(self):
        """sum_cls[TP_cls / (TP_cls + FP_cls)] / cls_num"""
        Precs = np.diag(self.confusion_matrix_curr) / (
                np.sum(self.confusion_matrix_curr, axis=0)) * 100.0
        Precs_sel = Precs[self.cls_sel_ind]
        MPrec = np.nanmean(Precs_sel)
        return MPrec, Precs

    def Mean_Recall(self):
        """sum_cls[TP_cls / (TP_cls + FN_cls)] / cls_num"""
        Recalls = np.diag(self.confusion_matrix_curr) / (
                np.sum(self.confusion_matrix_curr, axis=1)) * 100.0
        Recalls_sel = Recalls[self.cls_sel_ind]
        MRecall = np.nanmean(Recalls_sel)
        return MRecall, Recalls

    def Mean_F1_Score(self):
        """2 * (Prec * Recall) / (Prec + Recall)"""
        _, Precs = self.Mean_Precision()
        _, Recalls = self.Mean_Recall()
        F1_Scores = 2 * (Precs * Recalls) / (Precs + Recalls)
        F1_Scores_sel = F1_Scores[self.cls_sel_ind]
        MF1_Scores = np.nanmean(F1_Scores_sel)
        return MF1_Scores, F1_Scores

    def Mean_IoU(self):
        """sum_cls[TP_cls / (TP_cls + FN_cls + FP_cls)] / cls_num"""
        IoUs = np.diag(self.confusion_matrix_curr) / (
                np.sum(self.confusion_matrix_curr, axis=1) +
                np.sum(self.confusion_matrix_curr, axis=0) -
                np.diag(self.confusion_matrix_curr)) * 100.0
        IoUs_sel = IoUs[self.cls_sel_ind]
        MIoU = np.nanmean(IoUs_sel)
        return MIoU, IoUs

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix_curr, axis=1) / np.sum(self.confusion_matrix_curr)
        iu = np.diag(self.confusion_matrix_curr) / (
                np.sum(self.confusion_matrix_curr, axis=1) +
                np.sum(self.confusion_matrix_curr, axis=0) -
                np.diag(self.confusion_matrix_curr))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum() * 100.0
        return FWIoU

    def _generate_matrix(self, gt_image, pre_image):
        """absolute num confusion matrix: row-label_gt ; column-label_pred"""
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
        count = np.bincount(label, minlength=self.num_class**2)
        confusion_matrix = count.reshape(self.num_class, self.num_class)
        return confusion_matrix

    def cal_batch(self, target, pred):
        if isinstance(target, torch.Tensor):
            gt_image = target.data.cpu().numpy().astype(np.int64)
        else:
            gt_image = target.astype(np.int64)

        # get pred cls index
        if pred.shape[1] != 1:
            # net output
            _, pre_image = pred.topk(1, dim=1, largest=True, sorted=True)  # N,C,H,W => N,1,H,W
        else:
            # pred label
            pre_image = pred

        if isinstance(pre_image, torch.Tensor):
            pre_image = pre_image.data.cpu().numpy().astype(np.int64)
        else:
            pre_image = pre_image.astype(np.int64)

        # if target shape N,H,W
        if len(gt_image.shape) == 3:
            gt_image = gt_image.reshape(pre_image.shape[0], -1, pre_image.shape[-2], pre_image.shape[-1])  # N,H,W => N,1,H,W
        assert gt_image.shape == pre_image.shape

        self.confusion_matrix_curr = self._generate_matrix(gt_image, pre_image)
        self.confusion_matrix_all += self.confusion_matrix_curr

    def reset(self):
        self.confusion_matrix_all = np.zeros((self.num_class,) * 2)
