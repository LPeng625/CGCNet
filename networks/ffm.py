import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule, ModuleList, Sequential

import math


class Flatten(nn.Module):
    def forward(self, x):
        x_out = x.view(x.size(0), -1)
        return x_out


class ChannelGate(nn.Module):
    def __init__(self, channel, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = channel

        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channel, channel // reduction_ratio),
            nn.ReLU(),
            nn.Linear(channel // reduction_ratio, channel)
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid(channel_att_sum)
        scale = scale.unsqueeze(2)
        scale = scale.unsqueeze(3)
        scale = scale.expand_as(x)
        return x * scale


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale


class CSAM(nn.Module):
    def __init__(self, channel):
        super(CSAM, self).__init__()
        self.ChannelGate = ECAAPMP(channel)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class CBAM(nn.Module):
    def __init__(self, channel, reduction_ratio=16, pool_types=['avg', 'max']):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(channel, reduction_ratio, pool_types)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class ECAAP(nn.Module):

    def __init__(self, channel, gamma=2, b=1):
        super(ECAAP, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

class ECAMP(nn.Module):

    def __init__(self, channel, gamma=2, b=1):
        super(ECAMP, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        self.amp_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.amp_pool(x).view([b, 1, c])
        out = self.conv(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x


class ECAAPMP(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECAAPMP, self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.amp_pool = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, 1, c])
        amp = self.amp_pool(x).view([b, 1, c])
        out = torch.cat([avg, amp], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x

class FeatureFusionModule(BaseModule):
    """Feature Fusion Module. This module is different from FeatureFusionModule
    in BiSeNetV1. It uses two ConvModules in `self.attention` whose inter
    channel number is calculated by given `scale_factor`, while
    FeatureFusionModule in BiSeNetV1 only uses one ConvModule in
    `self.conv_atten`.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        scale_factor (int): The number of channel scale factor.
            Default: 4.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): The activation config for conv layers.
            Default: dict(type='ReLU').
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 out_channels,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        self.FFM_has_CSAM = True
        self.FFM_has_CBAM = False
        self.FFM_has_SE = False
        self.FFM_has_SAM = False
        self.FFM_has_ECA = False

        self.init_conv = ConvModule(
            2 * out_channels, out_channels, 1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        if self.FFM_has_CSAM:
            self.atten = CSAM(out_channels)

        if self.FFM_has_CBAM:
            self.atten = CBAM(out_channels)

        if self.FFM_has_SE:
            self.atten = SELayer(out_channels)

        if self.FFM_has_SAM:
            self.atten = SpatialGate()

        if self.FFM_has_ECA:
            self.atten = ECAAP(out_channels)

    def forward(self, spatial_inputs, context_inputs):

        inputs = torch.cat([spatial_inputs, context_inputs], dim=1)
        x = self.init_conv(inputs)
        x_atten = self.atten(x)
        return x_atten + x
