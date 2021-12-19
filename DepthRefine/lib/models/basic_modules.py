import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x


class ConvBnLeakyRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d, bn_eps=1e-5,
                 leaky_alpha=0.3, has_leaky_relu=True, inplace=True, has_bias=False):
        super(ConvBnLeakyRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = norm_layer(out_planes, eps=bn_eps)
        self.has_leakyrelu = has_leaky_relu
        if self.has_leakyrelu:
            self.relu = nn.LeakyReLU(negative_slope=leaky_alpha, inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_leakyrelu:
            x = self.relu(x)

        return x


class SeparableConvBnLeakyRelu(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, dilation=1,has_bn=True,inplace=True,
                 leaky_alpha=0.3, has_leaky_relu=True, norm_layer=nn.BatchNorm2d, has_bias=True):
        super(SeparableConvBnLeakyRelu, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding, dilation, groups=in_channels,
                               bias=has_bias)
        self.bn = norm_layer(in_channels)
        self.point_wise_cbr = ConvBnLeakyRelu(in_channels, out_channels, 1, 1, 0,
                                              has_bn=has_bn, norm_layer=norm_layer,
                                              leaky_alpha=leaky_alpha,
                                              has_leaky_relu=has_leaky_relu, inplace=inplace,
                                              has_bias=has_bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.point_wise_cbr(x)
        return x


class SeparableConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, padding=0, dilation=1,has_bn=True,inplace=True,
                 has_relu=True, norm_layer=nn.BatchNorm2d, has_bias=True):
        super(SeparableConvBnRelu, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride,
                               padding, dilation, groups=in_channels,
                               bias=has_bias)
        self.bn = norm_layer(in_channels)
        self.point_wise_cbr = ConvBnRelu(in_channels, out_channels, 1, 1, 0,
                                         has_bn=has_bn, norm_layer=norm_layer,
                                         has_relu=has_relu, inplace=inplace,
                                         has_bias=has_bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.point_wise_cbr(x)
        return x


class SELayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y


# For DFN
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, reduction):
        super(ChannelAttention, self).__init__()
        self.channel_attention = SELayer(in_planes, out_planes, reduction)

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], 1)
        channel_attetion = self.channel_attention(fm)
        fm = x1 * channel_attetion + x2

        return fm


class BNRefine(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, has_bias=False,
                 has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-5):
        super(BNRefine, self).__init__()
        self.conv_bn_relu = ConvBnRelu(in_planes, out_planes, ksize, 1,
                                       ksize // 2, has_bias=has_bias,
                                       norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize,
                                     stride=1, padding=ksize // 2, dilation=1,
                                     bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        t = self.conv_bn_relu(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


class RefineResidual(nn.Module):
    def __init__(self, in_planes, out_planes, relu_layer, ksize=3, has_bias=False,
                 has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-5, leaky_alpha=0.3, inplace=True):
        super(RefineResidual, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                  stride=1, padding=0, dilation=1,
                                  bias=has_bias)
        self.cbr = ConvBnRelu(out_planes, out_planes, ksize, 1,
                              ksize // 2, has_bias=has_bias,
                              norm_layer=norm_layer, bn_eps=bn_eps)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize,
                                     stride=1, padding=ksize // 2, dilation=1,
                                     bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            if relu_layer == 'ReLU':
                self.relu = nn.ReLU(inplace=inplace)
            elif relu_layer == 'LeakyReLU':
                self.relu = nn.LeakyReLU(negative_slope=leaky_alpha, inplace=inplace)

    def forward(self, x):
        x = self.conv_1x1(x)
        t = self.cbr(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


class SeparableRefineResidual(nn.Module):
    def __init__(self, in_planes, out_planes, relu_layer, ksize=3, has_bias=False,
                 has_relu=False, norm_layer=nn.BatchNorm2d, bn_eps=1e-5, leaky_alpha=0.3, inplace=True):
        super(SeparableRefineResidual, self).__init__()
        self.conv_1x1 = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                  stride=1, padding=0, dilation=1,
                                  bias=has_bias)
        self.cbr = SeparableConvBnRelu(out_planes, out_planes, ksize, 1,
                              ksize // 2, has_bias=has_bias,
                              norm_layer=norm_layer)
        self.conv_refine = nn.Conv2d(out_planes, out_planes, kernel_size=ksize,
                                     stride=1, padding=ksize // 2, dilation=1,
                                     bias=has_bias)
        self.has_relu = has_relu
        if self.has_relu:
            if relu_layer == 'ReLU':
                self.relu = nn.ReLU(inplace=inplace)
            elif relu_layer == 'LeakyReLU':
                self.relu = nn.LeakyReLU(negative_slope=leaky_alpha, inplace=inplace)

    def forward(self, x):
        x = self.conv_1x1(x)
        t = self.cbr(x)
        t = self.conv_refine(t)
        if self.has_relu:
            return self.relu(t + x)
        return t + x


# For BiSeNet
class AttentionRefinement(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d):
        super(AttentionRefinement, self).__init__()
        self.conv_3x3 = ConvBnRelu(in_planes, out_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes, 1, 1, 0,
                       has_bn=True, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        fm = self.conv_3x3(x)
        fm_se = self.channel_attention(fm)
        fm = fm * fm_se

        return fm


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        in_size = inputs.size()
        inputs = inputs.view((in_size[0], in_size[1], -1)).mean(dim=2)
        inputs = inputs.view(in_size[0], in_size[1], 1, 1)

        return inputs


class FeatureFusion(nn.Module):
    def __init__(self, in_planes, out_planes,
                 reduction=1, norm_layer=nn.BatchNorm2d):
        super(FeatureFusion, self).__init__()
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(out_planes, out_planes // reduction, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=True, has_bias=False),
            ConvBnRelu(out_planes // reduction, out_planes, 1, 1, 0,
                       has_bn=False, norm_layer=norm_layer,
                       has_relu=False, has_bias=False),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        fm = torch.cat([x1, x2], dim=1)
        fm = self.conv_1x1(fm)
        fm_se = self.channel_attention(fm)
        output = fm + fm * fm_se
        return output
