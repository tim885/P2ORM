# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------

from __future__ import print_function, division
import os
import logging
import time


def create_logger(root_output_path, args, image_set, curr_time, temp_file=False):
    cfg_path = args.config

    # set up logger
    if not os.path.exists(root_output_path):
        os.makedirs(root_output_path)
    assert os.path.exists(root_output_path), '{} does not exist'.format(
        root_output_path)

    cfg_name = os.path.basename(cfg_path).split('.')[0]
    if args.debug is True:  # debug mode, rm and create
        config_output_path = os.path.join(root_output_path, 'debug')
        cmd = "rm -r {0}".format(config_output_path)
        print('=>', cmd)
        os.system(cmd)
        os.makedirs(config_output_path)
    else:
        if (args.resume == '') or (args.out_dir != ''):  # new exp dir
            config_output_path = os.path.join(root_output_path, curr_time)
            if not os.path.exists(config_output_path):
                os.makedirs(config_output_path)
        else:  # save in resumed model dir
            exp_dir = args.resume.split('/')[0]
            config_output_path = os.path.join(root_output_path, exp_dir)

    # add image sets if needed
    image_sets = [iset for iset in image_set.split('+')]
    final_output_path = os.path.join(config_output_path,
                                     '{}'.format('_'.join(image_sets)))
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)

    if temp_file:
        log_file = 'temp_{}_{}.log'.format(cfg_name,
                                           time.strftime('%Y-%m-%d-%H-%M'))
    else:
        log_file = '{}_{}.log'.format(cfg_name,
                                      time.strftime('%Y-%m-%d-%H-%M'))
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(
        filename=os.path.join(final_output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger, final_output_path
