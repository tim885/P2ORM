# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------

from __future__ import print_function, division
import yaml
import numpy as np
from easydict import EasyDict as edict

# global config
config = edict()
config.pytorch_version = '1.1'
config.root_path = ''
config.output_path = ''

# default params
config.default = edict()
config.default.frequent = 20  # print log every frequent mini-batch

# ============================================= dataset related params =============================================== #
config.dataset = edict()
config.dataset.train_dataset = ''
config.dataset.val_dataset = ''
config.dataset.test_dataset = ''
config.dataset.image_set = ''
config.dataset.train_image_set = ''
config.dataset.val_image_set = ''
config.dataset.test_image_set = ''

# input transforms
config.dataset.input = 'image'

config.dataset.rand_blur_rate = 0.5  # blur image if rand num lower than thresh
config.dataset.color_jittering = True
config.dataset.norm_image = True  # normalize input image with means,stds
config.dataset.pixel_means = []
config.dataset.pixel_stds = []

# input/target co-transforms
config.dataset.minSize = 32  # net input should be multiple of this
config.dataset.base_size = 256  # base image size for random scale crop
config.dataset.crop_size = 256  # random crop size
config.dataset.test_fix_crop = []  # [H_min, H_max, W_min, W_max], fix crop model input in test
config.dataset.fill_label = 0
config.dataset.flip = False

# occlusion estimation params
config.dataset.val_w_mat_gt = False  # whether .mat gt exists on val set
config.dataset.save_w_rel_dir = False  # save res with relative dir in csv file

config.dataset.connectivity = 4  # neighborhood for occlusion relation estimation: [4,8]
config.dataset.NUM_TASKS = -1  # occ relation classification tasks num
config.dataset.TASK_CLASSES = -1  # occ relation classification tasks class num
config.dataset.NUM_CLASSES_H = -1  # class num along horizontal direction
config.dataset.NUM_CLASSES_V = -1  # class num along vertical direction
config.dataset.class_weights_path = ''
# ==================================================================================================================== #

# ============================================= network related params =============================================== #
config.network = edict()
config.network.arch = ''
config.network.model_prefix = ''
config.network.task_type = 'occ_order'  # ['occ_ori','occ_order']
config.network.pretrained = False
config.network.pretrained_path = ""  # pretrained model path, if not filled, get from web
config.network.init_type = ''
config.network.scale_down = 16  # downsampling factor, net input should be multiple of this
config.network.in_channels = -1
config.network.out_channels = -1

config.network.upsample = 'deconv'  # upsampling layer type: ['bilinear','deconv']
config.network.NUM_TASKS = -1  # detection tasks
config.network.out_activ = ''  # activation layer on the top of output layer
# ==================================================================================================================== #

# ============================================== params in train/val ================================================= #
config.TRAIN = edict()
config.TRAIN.optimizer = 'adam'  # ['sgd'|'adam']

# for SGD optimizer
config.TRAIN.momentum = 0.975
config.TRAIN.weight_decay = 0.0005
config.TRAIN.bias_decay = 0
config.TRAIN.nesterov = True

# for ADAM optimizer
config.TRAIN.betas = []

# learning rate schedule
config.TRAIN.warmup = False
config.TRAIN.warmup_lr = 0
config.TRAIN.warmup_step = 0
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 1000
config.TRAIN.lr = 0.0001
config.TRAIN.milestones = []

# loss variants
config.TRAIN.loss = ''
config.TRAIN.loss_gamma = []  # loss-term-wise weight
config.TRAIN.attentionloss_gamma_beta = []
config.TRAIN.smoothL1_sigma = 1.

config.TRAIN.class_weights = []  # class-balanced cross-entropy class-wise weight
config.TRAIN.spatial_weighting = []  # pixel-wise loss weight based on occlusion class
config.TRAIN.mask_is_edge = False  # use edge map as loss mask

# train set setup
config.TRAIN.batch_size = 64
config.TRAIN.shuffle = True
config.TRAIN.save_every_epoch = False  # save model
config.TRAIN.TENSORBOARD_LOG = True

# val set setup
config.TRAIN.val_metric = ''
config.TRAIN.val_step = 1  # val every step epochs
config.TRAIN.test_step = 9999  # test and save every step epochs
# ==================================================================================================================== #

# ================================================= params in test =================================================== #
config.TEST = edict()
config.TEST.batch_size = 1
config.TEST.img_padding = False  # only reflective padding to make img suitable for network
config.TEST.class_sel_ind = []  # selected class indices for occlusion evaluation
config.TEST.vis_type = ''
# ==================================================================================================================== #


def update_config(config_file):
    """update config with experiment-specific .yaml config file"""
    global config
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):  # when v is dict
                    if k == 'TRAIN':
                        if 'BBOX_WEIGHTS' in v:
                            v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
                    elif k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                    elif k == 'dataset':
                        if 'INTRINSIC_MATRIX' in v:
                            v['INTRINSIC_MATRIX'] = np.array(
                                v['INTRINSIC_MATRIX']).reshape([3, 3]).astype(
                                    np.float32)
                        if 'class_name_file' in v:
                            if v['class_name_file'] != '':
                                with open(v['class_name_file']) as f:
                                    v['class_name'] = [
                                        line.strip() for line in f.readlines()
                                    ]
                        if 'val_class_name_file' in v:
                            if v['val_class_name_file'] != '':
                                with open(v['val_class_name_file']) as f:
                                    v['val_class_name'] = [
                                        line.strip() for line in f.readlines()
                                    ]
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError(
                    "key: {} does not exist in config.py".format(k))
