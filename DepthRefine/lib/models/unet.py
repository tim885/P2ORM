import torch.nn as nn
import torch
from .basic_modules import ConvBnRelu, ConvBnLeakyRelu, RefineResidual


class UNet(nn.Module):
    def __init__(self, depth_channels=1, occ_channels=8, use_occ=True):
        super(UNet, self).__init__()
        self.use_occ = use_occ

        self.down_scale = nn.MaxPool2d(2)
        self.up_scale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        in_channels = depth_channels + occ_channels if use_occ else depth_channels

        # Encoder
        self.depth_down_layer0 = ConvBnLeakyRelu(in_channels, 32, 3, 1, 1, 1, 1,
                                                 has_bn=True, leaky_alpha=0.3,
                                                 has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_layer1 = ConvBnLeakyRelu(32, 64, 3, 1, 1, 1, 1,
                                                 has_bn=True, leaky_alpha=0.3,
                                                 has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_layer2 = ConvBnLeakyRelu(64, 128, 3, 1, 1, 1, 1,
                                                 has_bn=True, leaky_alpha=0.3,
                                                 has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_layer3 = ConvBnLeakyRelu(128, 256, 3, 1, 1, 1, 1,
                                                 has_bn=True, leaky_alpha=0.3,
                                                 has_leaky_relu=True, inplace=True, has_bias=True)
        self.depth_down_layer4 = ConvBnLeakyRelu(256, 256, 3, 1, 1, 1, 1,
                                                 has_bn=True, leaky_alpha=0.3,
                                                 has_leaky_relu=True, inplace=True, has_bias=True)

        # Decoder
        self.depth_up_layer0 = RefineResidual(256 * 2, 128, relu_layer='LeakyReLU',
                                              has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_layer1 = RefineResidual(128 * 2, 64, relu_layer='LeakyReLU',
                                              has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_layer2 = RefineResidual(64 * 2, 32, relu_layer='LeakyReLU',
                                              has_bias=True, has_relu=True, leaky_alpha=0.3)
        self.depth_up_layer3 = RefineResidual(32 * 2, 32, relu_layer='LeakyReLU',
                                              has_bias=True, has_relu=True, leaky_alpha=0.3)

        # Refiner
        self.refine_layer0 = ConvBnLeakyRelu(32 + in_channels, 16, 3, 1, 1, 1, 1,
                                             has_bn=True, leaky_alpha=0.3,
                                             has_leaky_relu=True, inplace=True, has_bias=True)
        self.refine_layer1 = ConvBnLeakyRelu(16, 10, 3, 1, 1, 1, 1,
                                             has_bn=True, leaky_alpha=0.3,
                                             has_leaky_relu=True, inplace=True, has_bias=True)

        self.output_layer = ConvBnRelu(10, 1, 3, 1, 1, 1, 1,
                                       has_bn=False, has_relu=False, inplace=True, has_bias=False)

    def forward(self, x, occ):
        m0 = torch.cat((x, occ[:, 1:, :, :]), 1) if self.use_occ else x

        #### Depth ####
        conv0 = self.depth_down_layer0(m0)
        x1 = self.down_scale(conv0)
        conv1 = self.depth_down_layer1(x1)
        x2 = self.down_scale(conv1)
        conv2 = self.depth_down_layer2(x2)
        x3 = self.down_scale(conv2)
        conv3 = self.depth_down_layer3(x3)
        x4 = self.down_scale(conv3)
        conv4 = self.depth_down_layer4(x4)

        #### Decode ####
        m4 = torch.cat((x4, conv4), 1)
        m4 = self.depth_up_layer0(m4)
        m3 = self.up_scale(m4)

        m3 = torch.cat((x3, m3), 1)
        m3 = self.depth_up_layer1(m3)
        m2 = self.up_scale(m3)

        m2 = torch.cat((x2, m2), 1)
        m2 = self.depth_up_layer2(m2)
        m1 = self.up_scale(m2)

        m1 = torch.cat((x1, m1), 1)
        m1 = self.depth_up_layer3(m1)
        m = self.up_scale(m1)

        ### Residual ###
        r = torch.cat((m0, m), 1)
        r = self.refine_layer0(r)
        r = self.refine_layer1(r)
        r = self.output_layer(r)

        x = (x + r).relu()
        return x
