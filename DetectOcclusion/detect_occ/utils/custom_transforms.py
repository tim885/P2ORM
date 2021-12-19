# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------

import random
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2
PI = 3.1416


# ------------------------------------- data pre-processing and augmentation ----------------------------------------- #
def get_data_transforms(config):
    """data transforms and augmentation for train/val"""
    # train input transform
    train_input_transf_list = []
    if config.dataset.color_jittering:
        train_input_transf_list += [transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.05)]
    train_input_transf = transforms.Compose(train_input_transf_list)

    # train input/target co-transforms
    train_co_transf_list = []
    if config.network.task_type == 'occ_order':
        if config.dataset.flip:
            train_co_transf_list += [OrderRandomHorizontalFlip(config)]
    elif config.network.task_type == 'occ_ori':
        train_co_transf_list += [OriRandomHorizontalFlip()]

    train_co_transf_list += [RandomCrop(config)]
    if config.dataset.input == 'image':
        if config.dataset.rand_blur_rate > 0:
            train_co_transf_list += [RandomGaussianBlur(prob_thresd=config.dataset.rand_blur_rate)]
        if config.dataset.norm_image:
            train_co_transf_list += [Normalize(mean=config.dataset.pixel_means, std=config.dataset.pixel_stds)]
    train_co_transf_list += [ToTensor(config)]
    train_co_transf = transforms.Compose(train_co_transf_list)

    # val input/target co-transforms
    val_co_transf_list = []
    if len(config.dataset.test_fix_crop) == 4:
        val_co_transf_list += [FixCrop(config, phase='test')]
    if config.dataset.norm_image:
        val_co_transf_list += [Normalize(mean=config.dataset.pixel_means, std=config.dataset.pixel_stds)]
    val_co_transf_list += [ToTensor(config)]
    val_co_transf = transforms.Compose(val_co_transf_list)

    return train_input_transf, train_co_transf, val_co_transf


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """return normalized numpy array"""
        img   = sample['image']
        masks = sample['label']  # list of mask

        img  = np.array(img)[:, :, :3].astype(np.float32)  # if RGBA => RGB
        img /= 255.0  # => [0,1]
        img -= self.mean
        img /= self.std

        for idx, _ in enumerate(masks):
            masks[idx] = np.array(masks[idx]).astype(np.float32)

        return {'image': img, 'label': masks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, config):
        self.config = config

    def __call__(self, sample):
        """
        :param sample: dict of PIL or numpy
        """
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        img   = sample['image']
        labels = sample['label']

        if isinstance(img, np.ndarray):
            # img = img.astype(np.float32)  # C,H,W
            img = img.astype(np.float32).transpose((2, 0, 1))  # H,W,C => C,H,W
        else:  # PIL image
            img = np.array(img).astype(np.float32).transpose((2, 0, 1))  # H,W,C => C,H,W
        if img.ndim == 2:
            img = torch.from_numpy(img).float().unsqueeze(0)  # H,W => 1,H,W
        else:
            img = torch.from_numpy(img).float()  # C,H,W

        if self.config.network.task_type == 'occ_order':
            for idx, _ in enumerate(labels):
                labels[idx] = np.array(labels[idx]).astype(np.float32)
                labels[idx] = torch.from_numpy(labels[idx]).long()
        elif self.config.network.task_type == 'occ_ori':
            occ_ori  = np.array(labels[0]).astype(np.float32)
            occ_edge = np.array(labels[1]).astype(np.float32)
            labels[0] = torch.from_numpy(occ_ori).float()
            labels[1] = torch.from_numpy(occ_edge).long()

        return {'image': img, 'label': labels}


class OrderRandomHorizontalFlip(object):
    """Random Horizontal Flip of P2ORM label map"""
    def __init__(self, config):
        super(OrderRandomHorizontalFlip, self).__init__()
        self.config = config

    def __call__(self, sample):
        """
        :param sample: 'image'; 'label':edge[0,1], order[0,1,2]
        """
        img   = sample['image']
        masks = sample['label']  # [label_E, label_S, label_SE, label_NE, label_edge]

        if random.random() < 0.5:
            if isinstance(img, np.ndarray):
                img = np.flip(img, axis=1)
            else:  # PIL image
                img = img.transpose(Image.FLIP_LEFT_RIGHT)

            # flip occ edge
            mask_edge      = masks[-1]
            mask_edge_flip = mask_edge.transpose(Image.FLIP_LEFT_RIGHT)
            masks[-1]      = mask_edge_flip

            if len(masks) != 1:
                # flip pixel pairwise occ order E,S
                mask_E      = masks[0]
                mask_E_flip = np.array(mask_E.transpose(Image.FLIP_LEFT_RIGHT))
                mask_E_flip[mask_E_flip == 0]   = 255
                mask_E_flip[mask_E_flip == 2]   = 0
                mask_E_flip[mask_E_flip == 255] = 2
                masks[0] = Image.fromarray(mask_E_flip)

                mask_S      = masks[1]
                mask_S_flip = mask_S.transpose(Image.FLIP_LEFT_RIGHT)
                masks[1]    = mask_S_flip

                if self.config.dataset.connectivity == 8:
                    # flip pixel pairwise occ order SE,NE
                    mask_SE = masks[2]
                    mask_NE = masks[3]
                    mask_SE_flip = np.array(mask_SE.transpose(Image.FLIP_LEFT_RIGHT))
                    mask_NE_flip = np.array(mask_NE.transpose(Image.FLIP_LEFT_RIGHT))
                    H, W = mask_SE_flip.shape

                    mask_SE_tmp           = np.ones([H + 1, W + 1])
                    mask_SE_tmp[:-1, :-1] = mask_NE_flip
                    mask_SE_tmp[mask_SE_tmp == 0]   = 255
                    mask_SE_tmp[mask_SE_tmp == 2]   = 0
                    mask_SE_tmp[mask_SE_tmp == 255] = 2
                    masks[2] = Image.fromarray(mask_SE_tmp[1:, 1:])

                    mask_NE_tmp          = np.ones([H+1, W+1])
                    mask_NE_tmp[1:, :-1] = mask_SE_flip
                    mask_NE_tmp[mask_NE_tmp == 0]   = 255
                    mask_NE_tmp[mask_NE_tmp == 2]   = 0
                    mask_NE_tmp[mask_NE_tmp == 255] = 2
                    masks[3] = Image.fromarray(mask_NE_tmp[:-1, 1:])

        return {'image': img, 'label': masks}


class OriRandomHorizontalFlip(object):
    """random horizontal flip occlusion edge&orientation"""
    def __init__(self):
        super(OriRandomHorizontalFlip, self).__init__()

    def __call__(self, sample):
        """
        :param sample: dict of PIL images
        :return:
        """
        img   = sample['image']
        masks = sample['label']  # [label_ori, label_edge]

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

            edge = np.array(masks[1])
            ori  = np.array(masks[0])

            edge = np.fliplr(edge)
            ori  = np.fliplr(ori)

            mask      = (edge == 1)
            ori[mask] = -ori[mask]

            masks[1] = Image.fromarray(edge)
            masks[0] = Image.fromarray(ori)

        return {'image': img, 'label': masks}


class RandomGaussianBlur(object):
    """derived from github pytorch-deeplab-xception/dataloaders/custom_transform"""
    def __init__(self, prob_thresd=0.5):
        self.prob_thresh = prob_thresd

    def __call__(self, sample):
        img   = sample['image']
        masks = sample['label']

        if random.random() < self.prob_thresh:
            img = img.filter(ImageFilter.GaussianBlur(radius=random.random()))

        return {'image': img, 'label': masks}


class FixCrop(object):
    def __init__(self, config, phase):
        self.config = config
        if phase == 'train':
            self.fix_crop = config.dataset.train_fix_crop  # [H_min, H_max, W_min, W_max]
        elif phase == 'test':
            self.fix_crop = config.dataset.test_fix_crop  # [H_min, H_max, W_min, W_max]
        else:
            raise ValueError('phase is not defined!')

    def __call__(self, sample):
        """
        fix crop input/label simultaneouly in train
        :param sample: {'image':, 'label':list}
        """
        img = sample['image']
        labels = sample['label']

        if isinstance(img, np.ndarray):
            img = img[self.fix_crop[0]:self.fix_crop[1], self.fix_crop[2]:self.fix_crop[3]]
        else:
            img = img.crop((self.fix_crop[2], self.fix_crop[0], self.fix_crop[3], self.fix_crop[1]))

        for idx, label in enumerate(labels):
            labels[idx] = label.crop((self.fix_crop[2], self.fix_crop[0], self.fix_crop[3], self.fix_crop[1]))

        return {'image': img, 'label': labels}


class RandomCrop(object):
    """
    Random crop with crop size
    derived from github pytorch-deeplab-xception/dataloaders/custom_transform
    """
    def __init__(self, config):
        self.config = config
        self.crop_size = config.dataset.crop_size
        self.edge_fill_lbl = 0
        self.ori_fill_lbl = 0
        self.order_fill_lbl = 1

    def __call__(self, sample):
        img = sample['image']
        labels = sample['label']

        if isinstance(img, np.ndarray):
            short_size = np.min(img.shape)
            oh, ow = img.shape
        else:  # PIL image
            short_size = np.min(img.size)
            ow, oh = img.size

        # pad before crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0

            if isinstance(img, np.ndarray):
                img = np.pad(img, ((0, padh), (0, padw)), 'constant', constant_values=0)
            else:  # PIL image
                img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)

            lbls_num = len(labels)
            for idx, _ in enumerate(labels):
                if idx < (lbls_num - 1):
                    if self.config.network.task_type == 'occ_order':  # pad label 1(no occ) for occ order
                        labels[idx] = ImageOps.expand(labels[idx], border=(0, 0, padw, padh), fill=self.order_fill_lbl)
                    elif self.config.network.task_type == 'occ_ori':  # pad label 0(no occ) for occ ori
                        labels[idx] = ImageOps.expand(labels[idx], border=(0, 0, padw, padh), fill=self.ori_fill_lbl)
                else:  # pad label 0(no occ) for occ edge
                    labels[idx] = ImageOps.expand(labels[idx], border=(0, 0, padw, padh), fill=self.edge_fill_lbl)

        # random crop with crop_size
        if isinstance(img, np.ndarray):
            h, w = img.shape
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
            img = img[y1:y1+self.crop_size, x1:x1+self.crop_size]
        else:
            w, h = img.size
            x1 = random.randint(0, w - self.crop_size)
            y1 = random.randint(0, h - self.crop_size)
            img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        for idx, _ in enumerate(labels):
            labels[idx] = labels[idx].crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img, 'label': labels}


# --------------------------------------------Data Conversion Tools -------------------------------------------------- #
def resize_to_origin(net_in, net_out, target, config):
    """
    resize net input/output to original image size offered by target
    :param net_in: tensor; N,3,H,W
    :param net_out: tensor; N,C,H,W
    :param target: tensor; N,H,W
    :return: 
    """
    N, C, H, W = net_out.shape
    _, H_org, W_org = target.shape

    if config.TEST.img_padding:
        # center crop if using reflect padding
        H_pad = int((H - H_org) / 2)
        W_pad = int((W - W_org) / 2)
        new_net_in = net_in[:, :, H_pad:H_pad + H_org, W_pad:W_pad + W_org]
        new_net_out = net_out[:, :, H_pad:H_pad+H_org, W_pad:W_pad+W_org]
    else:
        # resize model input
        mean_values = torch.tensor(config.dataset.pixel_means, dtype=net_in.dtype, device=net_in.device).view(3, 1, 1)
        std_values = torch.tensor(config.dataset.pixel_stds, dtype=net_in.dtype, device=net_in.device).view(3, 1, 1)
        img = net_in[0, :, :, :] * std_values + mean_values  # =>[0, 1]; 3,H,W

        img_np = tensor2im(img)  # [0,1] => [0,255] ; H,W,3
        img_np = resize_imgs([img_np], [H_org, W_org])[0]
        img_np = np.transpose(img_np, (2, 0, 1)) / 255.  # [0,255] => [0,1]; 3,H,W

        img = (torch.tensor(img_np, dtype=net_in.dtype, device=net_in.device) - mean_values) / std_values
        new_net_in = img.unsqueeze(0)  # 3,H,W => 1,3,H,W

        # resize outputs
        new_net_out = torch.zeros((1, C, H_org, W_org), dtype=net_out.dtype, device=net_out.device)
        if config.network.task_type == 'occ_order':
            for idx in range(0, C):
                out = net_out[0, idx, :, :]  # H,W
                out_np = out.cpu().float().numpy()
                out_np = cv2.resize(out_np, (W_org, H_org), interpolation=cv2.INTER_LANCZOS4)  # H,W

                out = torch.tensor(out_np, dtype=net_out.dtype, device=net_out.device)
                new_net_out[0, idx, :, :] = out
        elif config.network.task_type == 'occ_ori':
            out_edge = net_out[0, 0, :, :].cpu().float().numpy() * 255  # [0,1] => [0,255]
            out_ori  = net_out[0, 1, :, :].cpu().float().numpy()
            out_ori  = (out_ori.clip(-PI, PI) + PI) / PI / 2 * 255  # [-PI,PI] => [0,255]

            out_edge = np.array(Image.fromarray(out_edge.astype(np.uint8)).resize((W_org, H_org), Image.LANCZOS))
            out_edge = torch.tensor(out_edge, dtype=net_out.dtype, device=net_out.device) / 255  # [0,255] => [0,1]

            out_ori = np.array(Image.fromarray(out_ori.astype(np.uint8)).resize((W_org, H_org), Image.LANCZOS))
            out_ori = torch.tensor(out_ori, dtype=net_out.dtype, device=net_out.device)
            out_ori = out_ori / 255 * 2 * PI - PI  # [0,255] => [-PI,PI]

            new_net_out[0, 0, :, :] = out_edge
            new_net_out[0, 1, :, :] = out_ori

    return new_net_in, new_net_out


def tensor2im(image_tensor, bytes=255.0, imtype=np.uint8):
    """
    :param image_tensor: tensor, [0, 1]
    :param bytes:
    :param imtype:
    :return: numpy; H,W,C, [0, 255]
    """
    if image_tensor.dim() == 3:
        image_numpy = image_tensor.cpu().float().numpy()
    else:
        image_numpy = image_tensor[0].cpu().float().numpy()

    image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * bytes

    return image_numpy.astype(imtype)


def resize_imgs(imgs, hw, sampling_mode='LANCZOS'):
    """
    reisze img list to hw.
    :param imgs: list of numpy; [0, 255]; uint8;
    :return: list of numpy, [0, 255], uint8;
    """
    for idx, img in enumerate(imgs):
            img = img.squeeze()

            if sampling_mode == 'LANCZOS':
                img_tmp = np.array(Image.fromarray(img).resize((hw[1], hw[0]), Image.LANCZOS))
            elif sampling_mode == 'NEAREST':
                img_tmp = np.array(Image.fromarray(img).resize((hw[1], hw[0]), Image.NEAREST))

            if len(img_tmp.shape) == 2: img_tmp = img_tmp[:, :, np.newaxis]  # H,W => H,W,1
            imgs[idx] = img_tmp

    return imgs





