# --------------------------------------------------------
# P2ORM: Formulation, Inference & Application
# Licensed under The MIT License [see LICENSE for details]
# Written by Xuchong Qiu
# --------------------------------------------------------
# occlusion detection task model with resnet encoder from conv1 to conv4

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from .resnet import conv1x1, conv3x3, resnet18, resnet50
sys.path.append('..')
from detect_occ.utils.utility import kaiming_init, init_weights


def myResUnet(config, weights=None):
    """create UNet with given config"""
    model = ResUnetModel(config, encoder_arch='resnet50', pretrained=config.network.pretrained)

    if config.network.init_type == 'kaiming':
        model.apply(kaiming_init)
        print('=> use kaiming init')

    if weights is not None:
        model.load_state_dict(weights['state_dict'])

    return model


class ResUnetModel(nn.Module):
    def __init__(self, config, encoder_arch='resnet50', pretrained=True):
        """
        Unet arch with resnet encoder
        :param inplanes: num of encoder first conv out channel
        """
        super(ResUnetModel, self).__init__()
        self.config = config
        if encoder_arch == 'resnet50':
            self.encoder_out_ch = [64, 256, 512, 1024]  # to conv4
            self.decoder_out_ch = [1024, 512, 256, 64]
        elif encoder_arch == 'resnet18':
            self.encoder_out_ch = [64, 64, 128, 256]
            self.decoder_out_ch = [256, 128, 64, 64]

        # create encoder and load pretrained weights if needed
        self.inconv = DoubleConvBlock(in_nc=config.network.in_channels, mid_nc=64, out_nc=64, use_bias=True)

        # custom resnet encoder conv1
        if config.network.in_channels != 3:
            self.encode1 = nn.Sequential(
                nn.Conv2d(config.network.in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        if encoder_arch == 'resnet50':
            self.encoder = resnet50(pretrained=pretrained)
        elif encoder_arch == 'resnet18':
            self.encoder = resnet18(pretrained=pretrained)
        if pretrained:
            print('=> load imagenet pretrained weights')

        # bottleneck
        self.middle = DoubleConvBlock(in_nc=1024, mid_nc=512, out_nc=512, use_bias=True)

        # decoder
        self.deconv3 = UnetUpBlock(in_ch=1024, mid_ch=256, out_ch=256, config=config)
        self.deconv2 = UnetUpBlock(in_ch=512, mid_ch=64, out_ch=64, config=config)
        self.deconv1 = UnetUpBlock(in_ch=128, mid_ch=64, out_ch=64, config=config)
        self.deconv0 = UnetUpBlock(in_ch=128, mid_ch=64, out_ch=64, config=config)

        # output layers
        if self.config.network.task_type == 'occ_order':
            self.out0_order_E = OutputBlock(64, 3, kernel_size=1, use_bias=True, activation='None')
            self.out0_order_S = OutputBlock(64, 3, kernel_size=1, use_bias=True, activation='None')
            if self.config.dataset.connectivity == 8:
                self.out0_order_SE = OutputBlock(64, 3, kernel_size=1, use_bias=True, activation='None')
                self.out0_order_NE = OutputBlock(64, 3, kernel_size=1, use_bias=True, activation='None')
        elif self.config.network.task_type == 'occ_ori':
            self.out0_edge = OutputBlock(64, 1, kernel_size=1, use_bias=True, activation='sigmoid')
            self.out0_ori  = OutputBlock(64, 1, kernel_size=1, use_bias=True, activation='None')

    def forward(self, input0):
        """layer index indicate spatial size"""
        conv0 = self.inconv(input0)  # in_ch=>64
        if self.config.network.in_channels == 3:
            conv1, conv2, conv3, conv4 = self.encoder.forward(input0)  # 3=>64,256,512,1024
        else:
            conv1 = self.encode1(input0)  # in_ch=>64
            conv2, conv3, conv4 = self.encoder.forward_2to4(conv1)  # 64=>256,512,1024

        middle = self.middle(conv4)  # 1024=>512

        deconv3 = self.deconv3.forward(middle, conv3)  # (512+512)=>256
        deconv2 = self.deconv2.forward(deconv3, conv2)  # (256+256)=>64
        deconv1 = self.deconv1.forward(deconv2, conv1)  # (64+64)=>64
        deconv0 = self.deconv0.forward(deconv1, conv0)  # (64+64)=>64

        if self.config.network.task_type == 'occ_order':
            out0_E = self.out0_order_E(deconv0)  # 64 => 3
            out0_S = self.out0_order_S(deconv0)  # 64 => 3

            if self.config.dataset.connectivity == 4:
                return torch.cat((out0_E, out0_S), dim=1)
            elif self.config.dataset.connectivity == 8:
                out0_SE = self.out0_order_SE(deconv0)  # 64 => 3
                out0_NE = self.out0_order_NE(deconv0)  # 64 => 3
                return torch.cat((out0_E, out0_S, out0_SE, out0_NE), dim=1)

        elif self.config.network.task_type == 'occ_ori':
            out0_edge = self.out0_edge(deconv0)
            out0_ori  = self.out0_ori(deconv0)
            out0_edge_ori = torch.cat([out0_edge, out0_ori], dim=1)  # N,2,H,W

            return out0_edge_ori


class DoubleConvBlock(nn.Module):
    def __init__(self, in_nc, mid_nc, out_nc, use_bias=True):
        super(DoubleConvBlock, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_nc, mid_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(mid_nc),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_nc, out_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.BatchNorm2d(out_nc),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.model(x)


class ResBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, expansion=4):
        """
        derived from Resnet bottleneck class, with use-defined expansion
        :param inplanes: input ch num
        :param planes:
        :param stride: if !=1, downsample in conv2
        :param downsample:
        :param groups:
        :param base_width:
        :param dilation:
        :param norm_layer:
        :param expansion:
        """
        super(ResBottleneck, self).__init__()
        if norm_layer is None: norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.expansion = expansion  # ch num expansion ratio
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.downsample = downsample
        if self.stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1(inplanes, planes * self.expansion, self.stride),
                norm_layer(planes * self.expansion),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class UnetUpBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, config):
        super(UnetUpBlock, self).__init__()
        self.upsample = config.network.upsample
        if self.upsample == 'deconv':
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.double_conv = DoubleConvBlock(in_ch, mid_ch, out_ch)

    def forward(self, x1, x2):
        """upsample(x1) + x2 => double conv"""
        if self.upsample == 'deconv':
            x1 = self.up(x1, x2.shape[2:4])
        elif self.upsample == 'bilinear':
            x1 = F.interpolate(x1, size=x2.shape[2:4], mode='bilinear', align_corners=True)

        x = torch.cat([x2, x1], dim=1)
        x = self.double_conv(x)
        return x


class OutputBlock(nn.Module):
    def __init__(self, input_nc, output_nc, kernel_size=1, use_bias=True, activation='none'):
        super(OutputBlock, self).__init__()

        model = [
            nn.Conv2d(input_nc, output_nc, kernel_size=kernel_size, padding=0, bias=use_bias),
        ]

        if activation == 'tanh':
            model.append(nn.Tanh())  # [-1, 1]
        elif activation == 'sigmoid':
            model.append(nn.Sigmoid())  # [0, 1]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
