# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------
# define loss terms for occlusion edge/order detection

import torch
import torch.nn as nn
import torch.nn.functional as F
PI = 3.1416


def get_criterion(config, args):
    """get relevant loss for train"""
    if config.TRAIN.loss == 'CrossEntropyLoss':
        cls_weights = torch.tensor(config.TRAIN.class_weights, dtype=torch.float32)
        criterion = CCELoss(config, args.gpus[0], cls_weights=cls_weights,
                            spatial_weights=config.TRAIN.spatial_weighting, size_average=True)
    elif config.TRAIN.loss == 'AL_and_L1':
        criterion = AL_and_L1(config)
    return criterion


def cal_loss(criterion, net_out, targets, config):
    """
    :param net_out: N,C,H,W
    :param targets: occ order target [occ_order_1, ..., occ_order_C, occ_edge]
                    occ ori target   [occ_ori, occ_edge]
    """
    if config.network.task_type == 'occ_order':
        occ_order_loss_E = criterion(net_out[:, 0:3, :, :], targets[0], targets[-1])
        occ_order_loss_S = criterion(net_out[:, 3:6, :, :], targets[1], targets[-1])
        total_loss = (occ_order_loss_E + occ_order_loss_S)

        if config.dataset.connectivity == 4:
            return total_loss, [occ_order_loss_E, occ_order_loss_S]
        elif config.dataset.connectivity == 8:
            occ_order_loss_SE = criterion(net_out[:, 6:9, :, :], targets[2], targets[-1])
            occ_order_loss_NE = criterion(net_out[:, 9:12, :, :], targets[3], targets[-1])
            total_loss += (occ_order_loss_SE + occ_order_loss_NE)
            return total_loss, [occ_order_loss_E, occ_order_loss_S, occ_order_loss_SE, occ_order_loss_NE]

    elif config.network.task_type == 'occ_ori':
        occ_edge_loss, occ_ori_loss = criterion(net_out, targets)
        occ_edge_loss = config.TRAIN.loss_gamma[0] * occ_edge_loss
        occ_ori_loss  = config.TRAIN.loss_gamma[1] * occ_ori_loss
        total_loss = (occ_edge_loss + occ_ori_loss)

        return total_loss, [occ_edge_loss, occ_ori_loss]


class CCELoss(nn.Module):
    """class-balanced cross entropy loss for classification with given class weights"""
    def __init__(self, config, gpu_id, cls_weights='None', spatial_weights='None', size_average=True):
        super(CCELoss, self).__init__()
        self.config = config
        self.spatial_weights = spatial_weights
        self.size_average = size_average
        self.loss_CE = nn.CrossEntropyLoss(weight=cls_weights, reduction='none').cuda(gpu_id)

    def forward(self, net_out, target, mask):
        """
        :param net_out: N,C,H,W
        :param target: N,H,W; [0,1,2]
        :param mask: N,H,W; [0,1]
        :return:
        """
        loss = self.loss_CE(net_out, target)  # N,C,H,W => N,H,W
        loss = loss.view(-1)  # N,H,W => N*H*W,

        if self.spatial_weights != 'None':
            if self.config.TRAIN.mask_is_edge:  # where on occlusion edge pixels
                weight_mask = mask
            else:  # where curr occlusion order exists
                weight_mask = (target.clone().detach() != 1)

            weight_mask = weight_mask.view(-1).float()  # N,H,W => N*H*W,
            weight_mask[weight_mask == 1.] = self.spatial_weights[1]
            weight_mask[weight_mask == 0.] = self.spatial_weights[0]

            loss = weight_mask * loss

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class AL_and_L1(nn.Module):
    """Attention loss + SmoothL1 for occlusion edge/ori estimation"""
    def __init__(self, config):
        super(AL_and_L1, self).__init__()
        self.AttentionLoss = AttentionLoss(config.TRAIN.attentionloss_gamma_beta, avg_method='batch')
        self.SmoothL1Loss  = SmoothL1Loss(config.TRAIN.smoothL1_sigma, avg_method='batch')

    def forward(self, net_out, targets):
        occ_edge_pred = net_out[:, 0, :, :].unsqueeze(dim=1)  # N,1,H,W
        occ_ori_pred  = net_out[:, 1, :, :].unsqueeze(dim=1)
        occ_edge_gt = targets[1].unsqueeze(dim=1)
        occ_ori_gt  = targets[0].unsqueeze(dim=1)

        occ_edge_loss = self.AttentionLoss(occ_edge_pred, occ_edge_gt)
        occ_ori_loss  = self.SmoothL1Loss(occ_ori_pred, occ_ori_gt, occ_edge_gt)

        return occ_edge_loss, occ_ori_loss


class FocalLoss(nn.Module):
    """
    focal loss for classification task
    derived from https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
    """
    def __init__(self, gamma=0., alpha='None', size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma  # default:2 ; for RetinaNet: [0.5, 5]
        self.alpha = alpha  # hyper-param(list) or inverse class frequency
        if isinstance(alpha, (float, int)):  # binary case
            self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, net_out, target, weighting_map):
        cls_ind = torch.arange(0, net_out.size(1), dtype=torch.long).tolist()
        if net_out.dim() > 2:
            net_out = net_out.view(net_out.size(0), net_out.size(1), -1)  # N,C,H,W => N,C,H*W
            net_out = net_out.transpose(1, 2)  # N,C,H*W => N,H*W,C
            net_out = net_out.contiguous().view(-1, net_out.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)  # N,H,W => N*H*W,1

        log_pt = F.log_softmax(net_out, dim=1)  # log(softmax(net_out))
        log_pt = log_pt.gather(1, target)  # N*H*W,C => N*H*W,1 gather along axis 1
        log_pt = log_pt.view(-1)  # N*H*W,
        pt = log_pt.data.exp()  # softmax(net_out)

        if self.alpha == 'None':
            # use mini-batch inverse class frequency as alpha
            cls_num = torch.tensor([target.cpu().eq(cls_idx).sum() for cls_idx in cls_ind])
            # alpha = target.cpu().numel() / cls_num.float()  # C,
            alpha = 1 + torch.log(target.cpu().numel() / cls_num.float())  # C,
        else:
            alpha = self.alpha
        if alpha.type() != net_out.data.type():
            alpha = alpha.type_as(net_out.data)

        at = alpha.gather(0, target.data.view(-1))  # N*H*W,
        log_pt = log_pt * at  # alpha * log(softmax(net_out))
        loss = -1 * ((1 - pt) ** self.gamma) * log_pt  # N*H*W,
        if self.size_average:
            return loss.mean()  # average over each loss elem
        else:
            return loss.sum()


class AttentionLoss(nn.Module):
    """
    binary attention loss introduced in DOOBNet https://arxiv.org/pdf/1806.03772.pdf
    extension of focal loss by adding modulating param beta
    """

    def __init__(self, gamma_beta=(0.5, 4), alpha=None, size_average=True, avg_method='batch'):
        """
        :param gamma_beta:
        :param alpha: None or a float
        :param size_average:
        """
        super(AttentionLoss, self).__init__()
        self.gamma = gamma_beta[0]
        self.beta  = gamma_beta[1]
        self.alpha = alpha
        self.size_average = size_average
        self.avg_method   = avg_method

    def forward(self, net_out, target):
        """
        :param net_out:# net_out: (N, 1, H, W) ; activation passed by sigmoid [0~1]
        :param target: (N, 1, H, W)
        :return:
        """
        N, C, H, W = target.shape
        assert net_out.size(1) == 1

        # create mask to identify pixels at boundary
        edge     = (target == 1).float()
        non_edge = (target != 1).float()

        if self.alpha is None:
            alpha = non_edge.sum() / (non_edge.sum() + edge.sum())
        else:
            alpha = self.alpha

        net_out = torch.clamp(net_out, 0.00000001, 0.99999999)  # according to caffe code

        scale_edge    = alpha * torch.pow(self.beta, torch.pow((1 - net_out), self.gamma))
        scale_nonedge = (1 - alpha) * torch.pow(self.beta, torch.pow(net_out, self.gamma))

        log_p = net_out.log()
        log_m_p = (1 - net_out).log()

        loss = - edge * scale_edge * log_p - non_edge * scale_nonedge * log_m_p

        if self.size_average:
            if self.avg_method == 'batch':
                loss = loss.view(N, -1).sum(-1)
            return loss.mean()
        else:
            return loss.sum()  # too big loss may result in too big grad


class SmoothL1Loss(nn.Module):
    def __init__(self, sigma, avg_method):
        super(SmoothL1Loss, self).__init__()
        self.sigma = sigma
        self.avg_method = avg_method

    def forward(self, pred, gt, mask, size_average=True):
        """
        compute smoothL1 loss for orientation regression on given mask=1 pixels as in DOOBNet
        :param pred: N,1,H,W; float
        :param gt: N,1,H,W
        :param mask: N,1,H,W; [0, 1.]; int
        :param type:
        :return:
        """
        N, C, H, W = gt.shape
        valid_elem = (mask == 1.).sum()
        gt = gt.float()
        mask = mask.float()

        mask_sum_pos = (pred > PI) * (gt > 0.)
        mask_sum_neg = (pred < -PI) * (gt <= 0.)

        sum = pred + gt
        diff = pred - gt
        x = diff
        x[mask_sum_pos] = sum[mask_sum_pos]
        x[mask_sum_neg] = sum[mask_sum_neg]

        x = x * mask
        x_abs = torch.abs(x)

        mask_in  = (x_abs < (1 / self.sigma**2)).float()  # caffe code thresh: 1 / sigma**2; but thresh 1 works better
        mask_out = (x_abs >= (1 / self.sigma**2)).float()
        loss_in  = 0.5 * (self.sigma**2) * (x**2)
        loss_out = x_abs - 0.5 / (self.sigma**2)

        loss = loss_in * mask_in + loss_out * mask_out

        if size_average:
            if self.avg_method == 'batch':
                loss = torch.sum(loss.view(N, -1), dim=1)
                return loss.mean()
            elif self.avg_method == 'mean':
                return loss.sum() / valid_elem
        else:
            return loss.sum()


