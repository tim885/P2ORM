# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------

# Script for model inference without evaluation 

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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import sys
sys.path.append('..')
import models
from utils.utility import save_checkpoint, AverageMeter, kaiming_init
from utils.config import config, update_config
from utils.visualizer import viz_and_log, viz_and_save
from utils.custom_transforms import infer_data_transforms, resize_to_origin
from lib.logger.create_logger import create_logger
from lib.logger.print_and_log import print_and_log
from lib.dataset.occ_dataset import InferenceDataset

model_names = sorted(name for name in models.__dict__ if name.islower() and not name.startswith("__"))


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
best_perf = 0


def main():
    global args, curr_path, best_perf
    print('[INFO] Called with argument:', args)

    config.root_path = os.path.join(curr_path, '..')
    if args.out_dir:  # specify out dir
        output_path = os.path.join(config.root_path, config.output_path, args.out_dir)
    else:
        output_path = os.path.join(config.root_path, config.output_path)

    # logger
    curr_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    logger, save_path = create_logger(output_path, args, config.dataset.image_set,
                                      curr_time, temp_file=False)
    logger.info('[INFO] Called with args {}'.format(args))
    logger.info('[INFO] Called with config {}'.format(config))

    print_and_log('[INFO] will save everything to {}'.format(save_path), logger)
    if not os.path.exists(save_path): os.makedirs(save_path)

    # copy curr cfg file to output dir
    cmd = "cp {0} {1}/{2}_{3}.yaml".format(args.config, save_path,
                                           args.config.split('/')[-1].split('.')[0], curr_time)
    print('[INFO] Created current config file: {}_{}.yaml'
          .format(args.config.split('/')[-1].split('.')[0], curr_time))
    os.system(cmd)

    # create model
    network_data = None  # no pretrained weights
    args.arch = config.network.arch
    model = models.__dict__[args.arch](config, network_data)
    print_and_log("[INFO] Created model '{}'".format(args.arch), logger)

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
    print('[INFO] cudnn.deterministic flag:', cudnn.deterministic)

    # resume from a checkpoint with trained model
    model_path = os.path.join(output_path, args.resume)
    if os.path.isfile(model_path):
        print("[INFO] Loading resumed model '{}'".format(model_path))
        resumed_model = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(args.gpus[0]))
        model.load_state_dict(resumed_model['state_dict'])
        config.TRAIN.begin_epoch = resumed_model['epoch']
        best_perf    = resumed_model['best_perf']
        train_losses = resumed_model['train_losses']
        print("[INFO] Loaded resumed model '{}' (epoch {})".format(args.resume, resumed_model['epoch']))
        if config.TRAIN.end_epoch > len(train_losses):  # resume training with more epochs
            train_losses = np.append(train_losses, np.zeros([config.TRAIN.end_epoch - len(train_losses)]))
    else:
        print("[INFO] No checkpoint found at '{}'".format(args.resume))
        return

    # --------------------------------- inference on test set from resumed model ---------------------------------------- #
    print_and_log('[INFO] evaluation mode', logger)
    infer_csv = os.path.join(curr_path, '..', config.dataset.infer_image_set)
    input_transf = infer_data_transforms(config)
    infer_dataset = InferenceDataset(infer_csv, config, input_transf)
    infer_loader = torch.utils.data.DataLoader(
        infer_dataset, batch_size=1, num_workers=args.workers, pin_memory=True, shuffle=False
    )
    print_and_log('[INFO] {} test samples'.format(len(infer_dataset)), logger)

    infer_vis_writers = []
    for i in range(4):
        infer_vis_writers.append(SummaryWriter(os.path.join(save_path, 'test_vis', str(i))))
    results_vis_path = os.path.join(save_path, 'results_vis')
    print('[INFO] Save visual results to {}'.format(results_vis_path))
    if not os.path.exists(results_vis_path): os.makedirs(results_vis_path)

    inference(infer_loader, model, infer_vis_writers, logger, save_path, config)

    print_and_log('[INFO] Inference is finished.'.format(len(infer_dataset)), logger)

    return
    # ---------------------------------------------------------------------------------------------------------------- #


def inference(infer_loader, model, viz_writers, logger, res_path, config):
    global args
    batch_time = AverageMeter()
    data_time  = AverageMeter()

    batch_num = len(infer_loader)
    model.eval()
    with torch.no_grad():
        end = time.time()

        for batch_idx, (inputs, img_path, hw_org) in enumerate(tqdm(infer_loader)):
            data_time.update(time.time() - end)

            # load data
            net_in  = inputs.cuda(args.gpus[0], non_blocking=True)

            # forward
            net_out = model(net_in)

            batch_time.update(time.time() - end)
            end = time.time()

            # resize img and model output if needed
            net_in, net_out = resize_to_origin(net_in, net_out, hw_org, config)

            if batch_idx < len(viz_writers):  # visualize samples from first batches
                viz_and_log(net_in, net_out, targets=None, viz_writers=viz_writers, 
                            idx=batch_idx, epoch=0, config=config, w_target=False)

            # save every sample
            viz_and_save(net_in, net_out, img_path, res_path, config, epoch=0)

            # log curr batch info
            if batch_idx % config.default.frequent == 0:
                val_info = 'Val_Epoch: [{}][{}/{}]\t Time {}\t DataTime {}\t '\
                           .format(0, batch_idx, batch_num, batch_time, data_time)
                print_and_log(val_info, logger)

            if batch_idx >= batch_num: break
            if args.debug is True:
                if batch_idx == 20: break  # debug mode


if __name__ == '__main__':
    main()
