import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import atan, tan, pi
import itertools


def weights_normal_init(model, dev=0.001):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def kaiming_init(net):
    """Kaiming Init layer parameters."""
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0.2)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        elif isinstance(m, torch.nn.BatchNorm1d):
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)


def save_checkpoint(state, filename):
    torch.save(state, filename)
    print('save model at {}'.format(filename))


def load_checkpoint(model, optimizer, pth_file):
    print("loading checkpoint from {}".format(pth_file))
    checkpoint = torch.load(pth_file, map_location=lambda storage, loc: storage.cuda())
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])
    pretrained_dict = checkpoint['model']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Previous weight loaded')
    return epoch


def create_gamma_matrix(H=480, W=640, fx=600, fy=600):
    fov_x = 2 * atan(W / (2 * fx))
    fov_y = 2 * atan(H / (2 * fy))
    gamma = np.zeros((H, W, 2))

    for i, j in itertools.product(range(H), range(W)):
        alpha_x = (pi - fov_x) / 2
        gamma_x = alpha_x + fov_x * ((W - j) / W)

        alpha_y = (pi - fov_y) / 2
        gamma_y = alpha_y + fov_y * ((H - i) / H)

        gamma[i, j, 0] = gamma_x
        gamma[i, j, 1] = gamma_y

    return gamma


def huber_loss(pred, target, sigma, log=True):
    if log:
        pred_log = pred.clamp(1e-9).log()
        target_log = target.clamp(1e-9).log()
        diff_abs = torch.abs(pred_log - target_log)
    else:
        diff_abs = torch.abs(pred - target)
    cond = diff_abs < 1 / (sigma ** 2)
    loss = torch.where(cond, 0.5 * (sigma * diff_abs) ** 2, diff_abs - 0.5 / (sigma ** 2))
    return loss.mean()


def berhu_loss(pred, target, log=True):
    if log:
        pred_log = pred.clamp(1e-9).log()
        target_log = target.clamp(1e-9).log()
        diff_abs = (pred_log - target_log).abs()
    else:
        diff_abs = (pred - target).abs()
    delta = 0.2 * diff_abs.max()
    loss = torch.where(diff_abs < delta, diff_abs, (diff_abs ** 2 + delta ** 2) / (2 * delta + 1e-9))
    return loss.mean()


def spatial_gradient_loss(pred, target):
    sobel_x = torch.as_tensor([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]])

    sobel_x = sobel_x.view((1, 1, 3, 3)).type_as(pred)

    sobel_y = torch.as_tensor([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]])

    sobel_y = sobel_y.view((1, 1, 3, 3)).type_as(pred)

    pred_log = pred.clamp(1e-7).log()
    target_log = target.clamp(1e-7).log()

    diff = (pred_log - target_log)

    gx_diff = F.conv2d(diff, (1.0 / 8.0) * sobel_x, padding=1)
    gy_diff = F.conv2d(diff, (1.0 / 8.0) * sobel_y, padding=1)

    gradients_diff = torch.pow(gx_diff, 2) + torch.pow(gy_diff, 2)
    smooth_loss = gradients_diff.mean()

    gx_input = F.conv2d(pred_log, (1.0 / 8.0) * sobel_x, padding=1)
    gy_input = F.conv2d(pred_log, (1.0 / 8.0) * sobel_y, padding=1)

    gx_target = F.conv2d(target_log, (1.0 / 8.0) * sobel_x, padding=1)
    gy_target = F.conv2d(target_log, (1.0 / 8.0) * sobel_y, padding=1)

    gradients_input = torch.pow(gx_input, 2) + torch.pow(gy_input, 2)
    gradients_target = torch.pow(gx_target, 2) + torch.pow(gy_target, 2)

    grad_loss = huber_loss(gradients_input, gradients_target, 3, False)

    return smooth_loss + grad_loss


def neighbor_depth_variation(depth, diagonal=np.sqrt(2)):
    """Compute the variation of depth values in the neighborhood-8 of each pixel"""
    var1 = (depth[..., 1:-1, 1:-1] - depth[..., :-2, :-2]) / diagonal
    var2 = depth[..., 1:-1, 1:-1] - depth[..., :-2, 1:-1]
    var3 = (depth[..., 1:-1, 1:-1] - depth[..., :-2, 2:]) / diagonal
    var4 = depth[..., 1:-1, 1:-1] - depth[..., 1:-1, :-2]
    var6 = depth[..., 1:-1, 1:-1] - depth[..., 1:-1, 2:]
    var7 = (depth[..., 1:-1, 1:-1] - depth[..., 2:, :-2]) / diagonal
    var8 = depth[..., 1:-1, 1:-1] - depth[..., 2:, 1:-1]
    var9 = (depth[..., 1:-1, 1:-1] - depth[..., 2:, 2:]) / diagonal
    
    return torch.cat((var1, var2, var3, var4, var6, var7, var8, var9), 1)


def compute_tangent_adjusted_depth(depth_p, normal_p, depth_q, normal_q, eps=1e-3):
    # compute the depth map for the middl point
    depth_m = (depth_p + depth_q) / 2

    # compute the tangent-adjusted depth map for p and q
    ratio_p = (depth_p * normal_p).norm(dim=1, keepdim=True) / ((depth_m * normal_p).norm(dim=1, keepdim=True) + eps)
    depth_p_tangent = (depth_m * ratio_p).norm(dim=1, keepdim=True)

    ratio_q = (depth_q * normal_q).norm(dim=1, keepdim=True) / ((depth_m * normal_q).norm(dim=1, keepdim=True) + eps)
    depth_q_tangent = (depth_m * ratio_q).norm(dim=1, keepdim=True)

    return depth_p_tangent - depth_q_tangent


def neighbor_depth_variation_tangent(depth, normal, diagonal=np.sqrt(2)):
    """Compute the variation of tangent-adjusted depth values in the neighborhood-8 of each pixel"""
    depth_crop = depth[..., 1:-1, 1:-1]
    normal_crop = normal[..., 1:-1, 1:-1]
    var1 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., :-2, :-2], normal[..., :-2, :-2]) / diagonal
    var2 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., :-2, 1:-1], normal[..., :-2, 1:-1])
    var3 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., :-2, 2:], normal[..., :-2, 2:]) / diagonal
    var4 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., 1:-1, :-2], normal[..., 1:-1, :-2])
    var6 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., 1:-1, 2:], normal[..., 1:-1, 2:])
    var7 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., 2:, :-2], normal[..., 2:, :-2]) / diagonal
    var8 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., 2:, 1:-1], normal[..., 2:, 1:-1])
    var9 = compute_tangent_adjusted_depth(
        depth_crop, normal_crop, depth[..., 2:, 2:], normal[..., 2:, 2:]) / diagonal
    
    return torch.cat((var1, var2, var3, var4, var6, var7, var8, var9), 1)


def occlusion_aware_loss(depth_pred, occlusion, normal, gamma, th=1., diagonal=np.sqrt(2)):
    """
    Compute a distance between depth maps using the occlusion orientation
    :param depth_pred: (B, 1, H, W)
    :param occlusion: (B, 9, H, W)
    :param normal: (B, 3, H, W)
    :param gamma: (H, W, 2)
    """
    # change plane2plane depth map to point2point depth map
    delta_x = depth_pred / gamma[:, :, 0].tan()
    delta_y = depth_pred / gamma[:, :, 1].tan()
    depth_point = torch.cat((delta_x, delta_y, depth_pred), 1)

    # get neighborhood depth variation in (B, 8, H-2, W-2)
    depth_point_norm = depth_point.norm(dim=1, keepdim=True)
    depth_var_point = neighbor_depth_variation(depth_point_norm, diagonal)

    # get corrected neighborhood depth variation in (B, 8, H-2, W-2)
    depth_var_tangent = neighbor_depth_variation_tangent(depth_point, normal, diagonal)
    depth_var_min = torch.min(depth_var_tangent, depth_var_point)
    depth_var_geo = torch.where(depth_var_tangent > 0, depth_var_min, depth_var_point)

    # get masks in (B, 8, H-2, W-2)
    orientation = occlusion[:, 1:, 1:-1, 1:-1]

    # compute the loss for different cases
    th_tensor = torch.as_tensor(th).repeat(depth_var_geo.shape).type_as(depth_var_geo)

    fn_fg_mask = ((orientation == 1) * (depth_var_point > -th)).float()
    fn_bg_mask = ((orientation == -1) * (depth_var_point < th)).float()
    fp_fg_mask = ((orientation != 1) * (depth_var_geo < -th)).float()
    fp_bg_mask = ((orientation != -1) * (depth_var_geo > th)).float()

    fn_fg_loss = berhu_loss(depth_var_point[fn_fg_mask != 0], -th_tensor[fn_fg_mask != 0])
    fn_bg_loss = berhu_loss(depth_var_point[fn_bg_mask != 0], th_tensor[fn_bg_mask != 0])
    fp_fg_loss = berhu_loss(depth_var_geo[fp_fg_mask != 0], -th_tensor[fp_fg_mask != 0])
    fp_bg_loss = berhu_loss(depth_var_geo[fp_bg_mask != 0], th_tensor[fp_bg_mask != 0])

    loss_avg = fn_fg_loss + fn_bg_loss + fp_fg_loss + fp_bg_loss

    return loss_avg

