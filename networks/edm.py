import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
nonlinearity = partial(F.relu, inplace=True)
import math

class PDCConv2d(nn.Module):
    def __init__(self, pdc, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False):
        super(PDCConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.pdc = pdc

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):

        return self.pdc(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


## cd, ad, rd convolutions
def createConvFunc(op_type):

    assert op_type in ['cv', 'cd', 'ad', 'rd'], 'unknown op type: %s' % str(op_type)
    if op_type == 'cv':
        return op_type

    if op_type == 'cd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for cd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for cd_conv should be 3x3'
            assert padding == dilation, 'padding for cd_conv set wrong'

            weights_c = weights.sum(dim=[2, 3], keepdim=True)
            yc = F.conv2d(x, weights_c, stride=stride, padding=0, groups=groups)
            y = F.conv2d(x, weights, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y - yc
        return func
    elif op_type == 'ad':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for ad_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for ad_conv should be 3x3'
            assert padding == dilation, 'padding for ad_conv set wrong'

            shape = weights.shape
            weights = weights.view(shape[0], shape[1], -1)
            weights_conv = (weights - weights[:, :, [3, 0, 1, 6, 4, 2, 7, 8, 5]]).view(shape) # clock-wise
            y = F.conv2d(x, weights_conv, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    elif op_type == 'rd':
        def func(x, weights, bias=None, stride=1, padding=0, dilation=1, groups=1):
            assert dilation in [1, 2], 'dilation for rd_conv should be in 1 or 2'
            assert weights.size(2) == 3 and weights.size(3) == 3, 'kernel size for rd_conv should be 3x3'
            padding = 2 * dilation

            shape = weights.shape
            if weights.is_cuda:
                buffer = torch.cuda.FloatTensor(shape[0], shape[1], 5 * 5).fill_(0)
            else:
                buffer = torch.zeros(shape[0], shape[1], 5 * 5)
            weights = weights.view(shape[0], shape[1], -1)
            buffer[:, :, [0, 2, 4, 10, 14, 20, 22, 24]] = weights[:, :, 1:]
            buffer[:, :, [6, 7, 8, 11, 13, 16, 17, 18]] = -weights[:, :, 1:]
            buffer[:, :, 12] = 0
            buffer = buffer.view(shape[0], shape[1], 5, 5)
            y = F.conv2d(x, buffer, bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
            return y
        return func
    else:
        print('impossible to be here unless you force that')
        return None


nets = {
    'baseline': {
        'layer0':  'cv',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'cv',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'cv',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'c-v15': {
        'layer0':  'cd',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'cv',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'cv',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'a-v15': {
        'layer0':  'ad',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'cv',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'cv',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'r-v15': {
        'layer0':  'rd',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'cv',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'cv',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cv',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'cvvv4': {
        'layer0':  'cd',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'cd',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'cd',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'avvv4': {
        'layer0':  'ad',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'ad',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'ad',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'ad',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'rvvv4': {
        'layer0':  'rd',
        'layer1':  'cv',
        'layer2':  'cv',
        'layer3':  'cv',
        'layer4':  'rd',
        'layer5':  'cv',
        'layer6':  'cv',
        'layer7':  'cv',
        'layer8':  'rd',
        'layer9':  'cv',
        'layer10': 'cv',
        'layer11': 'cv',
        'layer12': 'rd',
        'layer13': 'cv',
        'layer14': 'cv',
        'layer15': 'cv',
        },
    'cccv4': {
        'layer0':  'cd',
        'layer1':  'cd',
        'layer2':  'cd',
        'layer3':  'cv',
        'layer4':  'cd',
        'layer5':  'cd',
        'layer6':  'cd',
        'layer7':  'cv',
        'layer8':  'cd',
        'layer9':  'cd',
        'layer10': 'cd',
        'layer11': 'cv',
        'layer12': 'cd',
        'layer13': 'cd',
        'layer14': 'cd',
        'layer15': 'cv',
        },
    'aaav4': {
        'layer0':  'ad',
        'layer1':  'ad',
        'layer2':  'ad',
        'layer3':  'cv',
        'layer4':  'ad',
        'layer5':  'ad',
        'layer6':  'ad',
        'layer7':  'cv',
        'layer8':  'ad',
        'layer9':  'ad',
        'layer10': 'ad',
        'layer11': 'cv',
        'layer12': 'ad',
        'layer13': 'ad',
        'layer14': 'ad',
        'layer15': 'cv',
        },
    'rrrv4': {
        'layer0':  'rd',
        'layer1':  'rd',
        'layer2':  'rd',
        'layer3':  'cv',
        'layer4':  'rd',
        'layer5':  'rd',
        'layer6':  'rd',
        'layer7':  'cv',
        'layer8':  'rd',
        'layer9':  'rd',
        'layer10': 'rd',
        'layer11': 'cv',
        'layer12': 'rd',
        'layer13': 'rd',
        'layer14': 'rd',
        'layer15': 'cv',
        },
    'c16': {
        'layer0':  'cd',
        'layer1':  'cd',
        'layer2':  'cd',
        'layer3':  'cd',
        'layer4':  'cd',
        'layer5':  'cd',
        'layer6':  'cd',
        'layer7':  'cd',
        'layer8':  'cd',
        'layer9':  'cd',
        'layer10': 'cd',
        'layer11': 'cd',
        'layer12': 'cd',
        'layer13': 'cd',
        'layer14': 'cd',
        'layer15': 'cd',
        },
    'a16': {
        'layer0':  'ad',
        'layer1':  'ad',
        'layer2':  'ad',
        'layer3':  'ad',
        'layer4':  'ad',
        'layer5':  'ad',
        'layer6':  'ad',
        'layer7':  'ad',
        'layer8':  'ad',
        'layer9':  'ad',
        'layer10': 'ad',
        'layer11': 'ad',
        'layer12': 'ad',
        'layer13': 'ad',
        'layer14': 'ad',
        'layer15': 'ad',
        },
    'r16': {
        'layer0':  'rd',
        'layer1':  'rd',
        'layer2':  'rd',
        'layer3':  'rd',
        'layer4':  'rd',
        'layer5':  'rd',
        'layer6':  'rd',
        'layer7':  'rd',
        'layer8':  'rd',
        'layer9':  'rd',
        'layer10': 'rd',
        'layer11': 'rd',
        'layer12': 'rd',
        'layer13': 'rd',
        'layer14': 'rd',
        'layer15': 'rd',
        },
    'carv4': {
        'layer0': 'cv',
        'layer1': 'cd',
        'layer2': 'ad',
        'layer3': 'rd',
        'layer4': 'cv',
        'layer5': 'cd',
        'layer6': 'ad',
        'layer7': 'rd',
        'layer8': 'cv',
        'layer9': 'cd',
        'layer10': 'ad',
        'layer11': 'rd',
        'layer12': 'cv',
        'layer13': 'cd',
        'layer14': 'ad',
        'layer15': 'rd',
    },
    }


def config_model(model):
    model_options = list(nets.keys())
    assert model in model_options, \
        'unrecognized RS3Mamba, please choose from %s' % str(model_options)

    print(str(nets[model]))

    pdcs = []
    for i in range(16):
        layer_name = 'layer%d' % i
        op = nets[model][layer_name]
        pdcs.append(createConvFunc(op))

    return pdcs

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bn=True, relu=True,
                 *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=False)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        self.relu = nn.ReLU() if relu else None
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class MapReduce(nn.Module):
    """
    Reduce feature maps into a single edge map
    """

    def __init__(self, channels):
        super(MapReduce, self).__init__()
        self.conv = nn.Conv2d(channels, 64, kernel_size=1, padding=0)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)

class PDCBlock(nn.Module):
    def __init__(self, pdc, inplane, ouplane, stride=1, is_ori=True):
        super(PDCBlock, self).__init__()
        self.stride = stride

        if self.stride > 1:
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.shortcut = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0)

        if (pdc == 'cv'):
            print('F.conv2d')
            self.conv1 = PDCConv2d(F.conv2d, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        else:
            self.conv1 = PDCConv2d(pdc, inplane, inplane, kernel_size=3, padding=1, groups=inplane, bias=False)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(inplane, ouplane, kernel_size=1, padding=0, bias=False)

        self.is_ori = is_ori
        if not self.is_ori:
            self.conv3 = nn.Conv2d(2 * inplane, ouplane, kernel_size=1, padding=0)

    def forward(self, x):
        if self.stride > 1:
            x = self.pool(x)
        y = self.conv1(x)
        y = self.relu2(y)
        if self.is_ori:
            y = self.conv2(y)
            if self.stride > 1:
                x = self.shortcut(x)
            y = y + x
        else:
            y = torch.cat((x, y), 1)
            y = self.conv3(y)
        return y

def init(str='carv4', is_ori=True, inplane=64):
    pdcs = config_model(str)
    return EDM(inplane=inplane, pdcs=pdcs, is_ori=is_ori)

class DSAM(nn.Module):

    def __init__(self, channel):
        super(DSAM, self).__init__()

        self.dilate1 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=1, dilation=1, bn=True, relu=True)
        self.dilate2 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=2, dilation=2, bn=True, relu=True)
        self.dilate3 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=4, dilation=4, bn=True, relu=True)
        self.dilate4 = ConvBNReLU(in_channels=channel, out_channels=channel, kernel_size=3,
                                  stride=1, padding=8, dilation=8, bn=True, relu=True)

        self.convInit = ConvBNReLU(in_channels=2 * channel, out_channels=channel, kernel_size=1,
                                   stride=1, padding=0, dilation=1, bn=True, relu=True)
        self.convFinal = ConvBNReLU(in_channels=5 * channel, out_channels=channel, kernel_size=1,
                                    stride=1, padding=0, dilation=1, bn=True, relu=True)
        self.SpatialGate = SpatialGate()

    def forward(self, x):
        x = self.convInit(x)
        dilate1_out = self.dilate1(x)
        dilate2_out = self.dilate2(dilate1_out)
        dilate3_out = self.dilate3(dilate2_out)
        dilate4_out = self.dilate4(dilate3_out)
        y = torch.cat([x, dilate1_out, dilate2_out, dilate3_out, dilate4_out], dim=1)
        out = self.convFinal(y)
        out = self.SpatialGate(out) + out

        return out

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = ConvBNReLU(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False, bn=True)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)
        return x * scale


class EDM(nn.Module):
    def __init__(self, inplane, pdcs, is_ori=True):
        super(EDM, self).__init__()

        self.fuseplanes = []
        self.inplane = inplane

        block_class = PDCBlock

        self.init_block = block_class(pdcs[0], 3, self.inplane, stride=2, is_ori=is_ori)
        self.block1_1 = block_class(pdcs[1], self.inplane, self.inplane, is_ori=is_ori)
        self.block1_2 = block_class(pdcs[2], self.inplane, self.inplane, is_ori=is_ori)
        self.block1_3 = block_class(pdcs[3], self.inplane, self.inplane, is_ori=is_ori)
        self.fuseplanes.append(self.inplane)  # C

        self.inplane = self.inplane * 2
        self.block2_1 = block_class(pdcs[4], inplane, self.inplane, stride=2, is_ori=is_ori)
        self.block2_2 = block_class(pdcs[5], self.inplane, self.inplane, is_ori=is_ori)
        self.block2_3 = block_class(pdcs[6], self.inplane, self.inplane, is_ori=is_ori)
        self.block2_4 = block_class(pdcs[7], self.inplane, self.inplane, is_ori=is_ori)
        self.fuseplanes.append(self.inplane)  # 2C

        self.conv_reduces = nn.ModuleList()

        for i in range(2):
            self.conv_reduces.append(MapReduce(self.fuseplanes[i]))


        self.dsam = DSAM(inplane)

    def forward(self, x):
        H, W = x.size()[2:]

        H, W = H // 2, W // 2

        x1 = self.init_block(x)
        x1 = self.block1_1(x1)
        x1 = self.block1_2(x1)
        x1 = self.block1_3(x1)

        x2 = self.block2_1(x1)
        x2 = self.block2_2(x2)
        x2 = self.block2_3(x2)
        x2 = self.block2_4(x2)

        e1 = self.conv_reduces[0](x1)
        e1 = F.interpolate(e1, (H, W), mode="bilinear", align_corners=False)
        e2 = self.conv_reduces[1](x2)
        e2 = F.interpolate(e2, (H, W), mode="bilinear", align_corners=False)

        out = torch.cat([e1, e2], dim=1)

        out = self.dsam(out)

        return out
