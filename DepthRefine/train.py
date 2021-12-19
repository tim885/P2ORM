import argparse
import os
import time

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from lib.models.unet import UNet
from lib.datasets.interior_net import InteriorNet

from lib.utils.net_utils import kaiming_init, weights_normal_init, save_checkpoint, load_checkpoint, \
    berhu_loss, spatial_gradient_loss, occlusion_aware_loss, create_gamma_matrix


# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

# network and loss settings
parser.add_argument('--use_occ', type=bool, default=True, help='whether to use occlusion as network input')
parser.add_argument('--th', type=float, default=15, help='depth discontinuity threshold in loss function (mm)')
parser.add_argument('--alpha_occ', type=float, default=0.01, help='weight balance')

# optimization settings
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate of optimizer')
parser.add_argument('--batch_size', type=int, default=8, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of output print')

# pth settings
parser.add_argument('--resume', action='store_true', help='resume checkpoint or not')
parser.add_argument('--checkpoint', type=str, default=None, help='optional reload model path')
parser.add_argument('--save_dir', type=str, default='save_model', help='save model path')
parser.add_argument('--dataset_dir', type=str, default='../data/InteriorNet_OR/data', help='training dataset')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# =================CREATE DATASET=========================== #
dataset_train = InteriorNet(opt.dataset_dir)
train_loader = DataLoader(dataset_train,
                          batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)
# ========================================================== #


# ================CREATE NETWORK AND OPTIMIZER============== #
net = UNet(use_occ=opt.use_occ)
net.apply(kaiming_init)
weights_normal_init(net.output_layer, 0.001)
net.cuda()

optimizer = optim.Adam(net.parameters(), lr=opt.lr)

if opt.resume:
    start_epoch = load_checkpoint(net, optimizer, opt.checkpoint) + 1
else:
    start_epoch = 1

gamma = create_gamma_matrix(480, 640, 600, 600)
gamma = torch.from_numpy(gamma).float().cuda()
# ========================================================== #


# =================== DEFINE TRAIN ========================= #
def train(data_loader, net, optimizer):
    net.train()
    end = time.time()
    for i, data in enumerate(data_loader):
        # load data and label
        depth_gt, depth_coarse, occlusion, normal, img = data
        depth_gt, depth_coarse, occlusion, normal, img = \
            depth_gt.cuda(), depth_coarse.cuda(), occlusion.cuda(), normal.cuda(), img.cuda()

        # forward pass
        depth_refined = net(depth_coarse, occlusion)

        # occlusion loss
        loss_depth_occ = occlusion_aware_loss(depth_refined, occlusion, normal, gamma, opt.th / 1000, 1)

        # regularization loss
        loss_change = berhu_loss(depth_refined, depth_coarse) + spatial_gradient_loss(depth_refined, depth_coarse)

        loss = opt.alpha_occ * loss_depth_occ + loss_change

        # optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure batch time
        batch_time = time.time() - end
        end = time.time()

        if i % opt.print_freq == 0:
            print("Epoch {} -- Iter [{}/{}] Occ loss: {:.3f}  Reg loss: {:.3f} || Batch time: {:.3f}".format(
                  epoch, i + 1, len(data_loader),
                  opt.alpha_occ * loss_depth_occ.item(),
                  loss_change.item(),
                  batch_time))
# ========================================================== #


if not os.path.exists(opt.save_dir):
    os.makedirs(opt.save_dir)

for epoch in range(start_epoch, opt.epoch + 1):

    train(train_loader, net, optimizer)

    save_checkpoint({
        'epoch': epoch,
        'model': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }, os.path.join(opt.save_dir, 'ckpt.pth'))
