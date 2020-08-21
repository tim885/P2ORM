# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------

from __future__ import print_function, division
import time
import argparse
import os
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from tqdm import tqdm
import sys
sys.path.append('..')
import models
from utils.utility import save_checkpoint, AverageMeter, kaiming_init
from utils.config import config, update_config
from utils.compute_loss import get_criterion, cal_loss
from utils.metrics import MetricsEvaluator
from utils.visualizer import viz_and_log, viz_and_save, plot_train_metrics, plot_val_metrics
from utils.custom_transforms import get_data_transforms, resize_to_origin
from lib.logger.create_logger import create_logger
from lib.logger.print_and_log import print_and_log
from lib.dataset.occ_dataset import ImgToOccDataset

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__"))


def parse_args():
    parser = argparse.ArgumentParser(description='Train and val for occlusion edge/order detection')
    parser.add_argument('--config', default='',
                        required=False, type=str, help='experiment configure file name')
    args, rest = parser.parse_known_args()
    update_config(args.config)  # update params with experiment config file

    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--new_val', action='store_true', help='new val with resumed model, re-calculate val perf ')
    parser.add_argument('--out_dir', default='', type=str, metavar='PATH', help='res output dir(defaut: output/date)')
    parser.add_argument('--evaluate', action='store_true', help='test with best model in validation')
    parser.add_argument('--frequent', default=config.default.frequent, type=int, help='frequency of logging')
    parser.add_argument('--gpus', help='specify the gpu to be use', default='3', required=False, type=str)
    parser.add_argument('--cpu', default=False, required=False, type=bool, help='whether use cpu mode')
    parser.add_argument('-j', '--workers', default=2, type=int, metavar='N', help='number of data loading workers')
    parser.add_argument('--vis', action='store_true', help='turn on visualization')
    parser.add_argument('--arch', '-a', metavar='ARCH', choices=model_names,
                        help='model architecture, overwritten if pretrained is specified: ' + ' | '.join(model_names))
    args = parser.parse_args()

    return args


args = parse_args()
curr_path = os.path.abspath(os.path.dirname(__file__))
n_iter = 0  # train iter num
best_perf = 0


def main():
    global args, curr_path, best_perf
    print('Called with argument:', args)

    config.root_path = os.path.join(curr_path, '..')
    if args.out_dir:  # specify out dir
        output_path = os.path.join(config.root_path, config.output_path, args.out_dir)
    else:
        output_path = os.path.join(config.root_path, config.output_path)

    # logger
    curr_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    logger, save_path = create_logger(output_path, args, config.dataset.image_set,
                                      curr_time, temp_file=False)
    logger.info('called with args {}'.format(args))
    logger.info('called with config {}'.format(config))

    print_and_log('=> will save everything to {}'.format(save_path), logger)
    if not os.path.exists(save_path): os.makedirs(save_path)

    # copy curr cfg file to output dir
    cmd = "cp {0} {1}/{2}_{3}.yaml".format(args.config, save_path,
                                           args.config.split('/')[-1].split('.')[0], curr_time)
    print('=> created current config file: {}_{}.yaml'
          .format(args.config.split('/')[-1].split('.')[0], curr_time))
    os.system(cmd)

    # set tensorboard writer
    train_writer = SummaryWriter(os.path.join(save_path, 'train_vis'))
    val_writer   = SummaryWriter(os.path.join(save_path, 'val_vis'))
    train_vis_writers = []
    val_vis_writers = []

    for i in range(4):  # writers for n sample visual res
        train_vis_writers.append(SummaryWriter(os.path.join(save_path, 'train_vis', str(i))))
        val_vis_writers.append(SummaryWriter(os.path.join(save_path, 'val_vis', str(i))))

    # create model
    network_data = None  # no pretrained weights
    args.arch = config.network.arch
    model = models.__dict__[args.arch](config, network_data)
    print_and_log("=> created model '{}'".format(args.arch), logger)

    # gpu settings
    gpu_ids = list(args.gpus.replace(',', ''))
    args.gpus = [int(gpu_id) for gpu_id in gpu_ids]

    if args.gpus.__len__() == 1:
        # single GPU
        torch.cuda.set_device(args.gpus[0])
        model = model.cuda(args.gpus[0])
    else:
        model = torch.nn.DataParallel(model, device_ids=args.gpus)

    cudnn.benchmark = False  # True if fixed train/val input size
    print('=> cudnn.deterministic flag:', cudnn.deterministic)

    criterion = get_criterion(config, args)  # define loss functions(criterion)

    # define optimizer and lr scheduler
    assert (config.TRAIN.optimizer in ['adam', 'sgd'])
    print_and_log('=> setting {} solver'.format(config.TRAIN.optimizer), logger)
    if config.TRAIN.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), config.TRAIN.lr, betas=config.TRAIN.betas)
    elif config.TRAIN.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), config.TRAIN.lr, momentum=config.TRAIN.momentum,
                                    weight_decay=config.TRAIN.weight_decay, nesterov=config.TRAIN.nesterov)

    if args.resume:  # optionally resume from a checkpoint with trained model or train from scratch
        model_path = os.path.join(output_path, args.resume)
        if os.path.isfile(model_path):
            print("=> loading resumed model '{}'".format(model_path))
            resumed_model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(args.gpus[0]))
            model.load_state_dict(resumed_model['state_dict'])
            optimizer.load_state_dict(resumed_model['optimizer'])
            config.TRAIN.begin_epoch = resumed_model['epoch']
            best_perf    = resumed_model['best_perf']
            train_losses = resumed_model['train_losses']
            date_time    = resumed_model['date_time']  # date of beginning
            print("=> loaded resumed model '{}' (epoch {})".format(args.resume, resumed_model['epoch']))
            if config.TRAIN.end_epoch > len(train_losses):  # resume training with more epochs
                train_losses = np.append(train_losses, np.zeros([config.TRAIN.end_epoch - len(train_losses)]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return
    else:
        train_losses = np.zeros(config.TRAIN.end_epoch)
        date_time = curr_time

    # setup learning schedule
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.TRAIN.milestones, gamma=0.1)
    for epoch in range(0, config.TRAIN.begin_epoch):
        lr_scheduler.step()

    # --------------------------------- test on test set from resumed model ------------------------------------------ #
    if args.evaluate:
        print_and_log('=> evaluation mode', logger)
        test_csv = os.path.join(curr_path, '..', config.dataset.test_image_set)
        _, _, test_co_transf = get_data_transforms(config)
        test_dataset = ImgToOccDataset(test_csv, config, True, None, None, test_co_transf)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=args.workers,
                                                  pin_memory=True, shuffle=False)
        print_and_log('=> {} test samples'.format(len(test_dataset)), logger)

        epoch = config.TRAIN.begin_epoch
        test_vis_writers = []
        for i in range(4):
            test_vis_writers.append(SummaryWriter(os.path.join(save_path, 'test_vis', str(i))))
        results_vis_path = os.path.join(save_path, 'results_vis')
        print('save visual results to {}'.format(results_vis_path))
        if not os.path.exists(results_vis_path): os.makedirs(results_vis_path)

        _, val_mIoUs, val_mF1s, val_AP_edge = val_epoch(test_loader, model, criterion, epoch, test_vis_writers,
                                                        logger, save_path, config, isTest=True)

        if config.network.task_type == 'occ_order':
            test_info = '[Test] meanIoU_E:{}, meanIoU_S:{}, meanF1_E:{}, meanF1_S:{} '\
                        .format(val_mIoUs[0], val_mIoUs[1], val_mF1s[0], val_mF1s[0])
            if config.dataset.connectivity == 8:
                test_info += '[Test] meanIoU_SE:{}, meanIoU_NE:{}, meanF1_SE:{}, meanF1_NE:{} ' \
                            .format(val_mIoUs[2], val_mIoUs[3], val_mF1s[2], val_mF1s[3])
        elif config.network.task_type == 'occ_ori':
            test_info = '[Test] meanF1_edge:{}, AP_edge:{} '.format(val_mF1s[0], val_AP_edge)

        val_mIoU = sum(val_mIoUs) / float(len(val_mIoUs))
        test_info += 'meanIoU_all:{}'.format(val_mIoU)
        print_and_log(test_info, logger)

        return
    # ---------------------------------------------------------------------------------------------------------------- #

    # create dataloaders for train/val
    train_csv = os.path.join(curr_path, '..', config.dataset.train_image_set)
    val_csv = os.path.join(curr_path, '..', config.dataset.val_image_set)
    print_and_log("=> train_image_set: {}\n=> val_image_set: {}"
                  .format(config.dataset.train_image_set, config.dataset.val_image_set), logger)

    train_input_transf, train_co_transf, val_co_transf = get_data_transforms(config)
    train_dataset = ImgToOccDataset(train_csv, config, False, train_input_transf, None, train_co_transf)
    val_dataset = ImgToOccDataset(val_csv, config, True, None, None, val_co_transf)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN.batch_size,
                                               num_workers=args.workers, pin_memory=True,
                                               shuffle=config.TRAIN.shuffle)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.TEST.batch_size,
                                             num_workers=args.workers, pin_memory=True, shuffle=False)
    print_and_log('=> {} samples found, {} train samples and {} validation samples'
                  .format(len(val_dataset) + len(train_dataset), len(train_dataset), len(val_dataset)), logger)

    # ----------------------------------------------- training loop -------------------------------------------------- #
    val_step = config.TRAIN.val_step  # val every step epochs
    for epoch in tqdm(range(config.TRAIN.begin_epoch, config.TRAIN.end_epoch)):
        train_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # train on train set
        train_loss, train_mIoUs, train_mF1s, train_AP_edge = train_epoch(train_loader, model, criterion,
                                                                         optimizer, epoch, train_writer,
                                                                         train_vis_writers, config, logger)
        plot_train_metrics(train_writer, config, epoch, train_loss, train_mIoUs, train_mF1s, train_AP_edge)
        train_losses[epoch] = train_loss

        # evaluate on validation set
        is_best = False
        if (epoch + 1) % val_step == 0 or args.debug:
            val_loss, val_mIoUs, val_mF1s, val_AP_edge = val_epoch(val_loader, model, criterion,
                                                                   epoch, val_vis_writers,
                                                                   logger, save_path, config)
            plot_val_metrics(val_writer, config, epoch, val_loss, val_mIoUs, val_mF1s, val_AP_edge)

            # compare perf and save best model
            if config.TRAIN.val_metric == 'mIoU':
                val_mIoU = sum(val_mIoUs) / float(len(val_mIoUs))
                val_perf = val_mIoU
            elif config.TRAIN.val_metric == 'AP':
                val_perf = val_AP_edge
            else:
                raise ValueError("val metric not recognized.")

            if args.new_val and (epoch == config.TRAIN.begin_epoch):
                best_perf = 0  # new val, re-cal val perf

            is_best = val_perf > best_perf
            best_perf = max(val_perf, best_perf)

        # save checkpoint, if best, save best model
        model_dict = {'date_time': date_time, 'best_perf': best_perf, 'epoch': epoch,
                      'arch': args.arch, 'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict(), 'train_losses': train_losses}
        save_checkpoint(model_dict, is_best, save_path, config.TRAIN.save_every_epoch)

        lr_scheduler.step()
    # ---------------------------------------------------------------------------------------------------------------- #

    # close tensorboardX writers
    train_writer.close()
    val_writer.close()
    for i in range(4):
        train_vis_writers[i].close()
        val_vis_writers[i].close()


def train_epoch(train_loader, model, criterion, optimizer, epoch, train_writer, output_writers, config, logger):
    global n_iter, args
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    train_eval = MetricsEvaluator(config, isTrain=True)

    batch_num = len(train_loader)
    model.train()
    end = time.time()

    info = 'Train_Epoch: [{}] current learning rate: {}'.format(epoch, optimizer.param_groups[0]['lr'])
    print_and_log(info, logger)

    for batch_idx, (inputs, targets, img_path) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # load data
        net_in  = inputs.cuda(args.gpus[0], non_blocking=True)
        targets = [target.cuda(args.gpus[0], non_blocking=True) for target in targets]

        # forward
        net_out = model(net_in)

        # cal loss and backward
        total_loss, loss_list = cal_loss(criterion, net_out, targets, config)
        losses.update(total_loss.item(), targets[0].size(0))
        train_writer.add_scalar('train_loss_iter', total_loss.item(), n_iter)
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # cal batch metrics
        train_eval.cal_batch_metrics(net_out, targets)

        if batch_idx < len(output_writers):  # visualize samples from first batches
            viz_and_log(net_in, net_out, targets, output_writers, batch_idx, epoch, config)

        # log curr batch info
        if batch_idx % config.default.frequent == 0:
            train_info = 'Train_Epoch: [{}][{}/{}]\t Time {}\t DataTime {}\t '\
                .format(epoch, batch_idx, batch_num, batch_time, data_time)
            if config.network.task_type == 'occ_order':
                train_info += 'TotalLoss {}\t Loss_order_E {:.5f}\t Loss_order_S {:.5f}\t ' \
                              'meanIoU_E {:.3f}\t meanIoU_S {:.3f}\t '\
                    .format(losses, loss_list[0], loss_list[1],
                            train_eval.curr_mIoU_E, train_eval.curr_mIoU_S)
                if config.dataset.connectivity == 8:
                    train_info += 'Loss_order_SE {:.5f}\t Loss_order_NE {:.5f}\t ' \
                                  'meanIoU_SE {:.3f}\t meanIoU_NE {:.3f}'\
                            .format(loss_list[2], loss_list[3],
                                    train_eval.curr_mIoU_SE, train_eval.curr_mIoU_NE)
            elif config.network.task_type == 'occ_ori':
                train_info += 'TotalLoss {}\t Loss_edge {:.5f}\t Loss_ori {:.5f}\t ' \
                              'meanF1 {:.3f}\t meanAP {:.3f}\t ' \
                    .format(losses, loss_list[0], loss_list[1],
                            train_eval.curr_mF1_edge, train_eval.curr_AP_edge)
            print_and_log(train_info, logger)

        n_iter += 1
        if batch_idx >= batch_num: break
        if args.debug is True:
            if batch_idx == 20: break  # debug mode

    # log avg perf over dataset
    train_eval.cal_set_metrics()
    print_and_log('Train_Epoch: [{}]\t Average meanIoU: {:.3f}\t Average meanF1: {:.3f}'
                  .format(epoch, train_eval.mIoU_all, train_eval.mF1_all), logger)

    return losses.avg, train_eval.mIoUs, train_eval.mF1s, train_eval.AP_edge.avg


def val_epoch(val_loader, model, criterion, epoch, output_writers, logger, res_path, config, isTest=False):
    global args
    batch_time = AverageMeter()
    data_time  = AverageMeter()
    losses     = AverageMeter()
    val_eval = MetricsEvaluator(config, isTrain=False)

    batch_num = len(val_loader)
    model.eval()
    with torch.no_grad():
        end = time.time()

        for batch_idx, (inputs, targets, img_path) in enumerate(val_loader):
            data_time.update(time.time() - end)

            # load data
            net_in  = inputs.cuda(args.gpus[0], non_blocking=True)
            targets = [target.cuda(args.gpus[0], non_blocking=True) for target in targets]

            # forward
            net_out = model(net_in)

            batch_time.update(time.time() - end)
            end = time.time()

            # resize img and model output if needed
            net_in, net_out = resize_to_origin(net_in, net_out, targets[0], config)

            # cal loss
            total_loss, loss_list = cal_loss(criterion, net_out, targets, config)
            losses.update(total_loss.item(), targets[0].size(0))

            # compute batch metrics
            val_eval.cal_batch_metrics(net_out, targets)

            if batch_idx < len(output_writers):  # visualize samples from first batches
                viz_and_log(net_in, net_out, targets, output_writers, batch_idx, epoch, config)

            # save every sample
            if isTest or (epoch + 1) % config.TRAIN.test_step == 0:
                out_path = os.path.join(res_path, 'results_vis')
                viz_and_save(net_in, net_out, img_path, out_path, config, epoch)

            # log curr batch info
            if batch_idx % config.default.frequent == 0:
                val_info = 'Val_Epoch: [{}][{}/{}]\t Time {}\t DataTime {}\t '\
                           .format(epoch, batch_idx, batch_num, batch_time, data_time)
                if config.network.task_type == 'occ_order':
                    val_info += 'TotalLoss {}\t Loss_order_E {:.5f}\t Loss_order_S {:.5f}\t ' \
                                'meanIoU_E {:.3f}\t meanIoU_S {:.3f}' \
                                .format(losses, loss_list[0], loss_list[1],
                                        val_eval.curr_mIoU_E, val_eval.curr_mIoU_S)
                    if config.dataset.connectivity == 8:
                        val_info += 'Loss_order_SE {:.5f}\t Loss_order_NE {:.5f}\t ' \
                                    'meanIoU_SE {:.3f}\t meanIoU_NE {:.3f}'\
                                    .format(loss_list[2], loss_list[3],
                                            val_eval.curr_mIoU_SE, val_eval.curr_mIoU_NE)
                elif config.network.task_type == 'occ_ori':
                    val_info += 'TotalLoss {}\t Loss_edge {:.5f}\t Loss_ori {:.5f}\t ' \
                                  'meanF1 {:.3f}\t meanAP {:.3f}\t ' \
                                .format(losses, loss_list[0], loss_list[1],
                                        val_eval.curr_mF1_edge, val_eval.curr_AP_edge)
                print_and_log(val_info, logger)

            if batch_idx >= batch_num: break
            if args.debug is True:
                if batch_idx == 20: break  # debug mode

    # cal avg perf over val set
    val_eval.cal_set_metrics(isTest=isTest)
    log_info = 'Val_Epoch: [{}]\t Avg mIoU: {:.3f}; Avg mF1: {:.3f}, Avg_AP_edge: {:.3f}'\
               .format(epoch, val_eval.mIoU_all, val_eval.mF1_all, val_eval.AP_edge.avg)
    print_and_log(log_info, logger)

    return losses.avg, val_eval.mIoUs, val_eval.mF1s, val_eval.AP_edge.avg


if __name__ == '__main__':
    main()
