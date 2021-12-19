# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------

import os
import shutil
import numpy as np
import torch
from torch.nn import init


def save_checkpoint(state, is_best, save_path, save_every_epoch=False):
    curr_model_path = os.path.join(save_path, 'checkpoint_{}.pth.tar'.format(state['epoch']))

    # remove last checkpoint model
    if not save_every_epoch:
        cmd = "rm {}/checkpoint_{}.pth.tar".format(save_path, state['epoch'] - 1)
        print('=>', cmd)
        os.system(cmd)

    torch.save(state, curr_model_path)
    print('=> save current epoch model', curr_model_path)  

    if is_best:
        # remove current best model
        cmd = "rm {}/model_best_*".format(save_path)
        print('=>', cmd)
        os.system(cmd)

        best_model_path = os.path.join(save_path, 'model_best_{}.pth.tar'.format(state['epoch']))
        shutil.copyfile(curr_model_path, best_model_path)
        print('=> save best model', best_model_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        if not np.isnan(val):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def kaiming_init(net):
    """Kaiming Init layer parameters."""
    for m in net.modules():
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, a=0., mode='fan_in')  # custom: a=0.2
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)

        elif isinstance(m, torch.nn.BatchNorm2d):
            torch.nn.init.constant_(m.bias, 0)
            torch.nn.init.constant_(m.weight, 1)  # or uniform

        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)


def init_weights(net, init_type='normal', gain=0.02):
    """
    Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

        elif classname.find('BatchNorm2d') != -1:
            init.uniform_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


