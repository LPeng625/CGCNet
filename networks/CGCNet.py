import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from functools import partial
nonlinearity = partial(F.relu, inplace=True)



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class NonLocalBlock(nn.Module):
    def __init__(self, channel, size=(32, 32)):
        super(NonLocalBlock, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                  padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                    padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=channel, kernel_size=1, stride=1,
                                   padding=0, bias=False)

        self.with_pos = True
        if self.with_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, size[0] * size[1], size[0] * size[1]))

        self.with_d_pos = True
        if self.with_d_pos:
            self.pos_embedding_decoder = nn.Parameter(torch.randn(1, self.inter_channel,
                                                              size[0], size[1]))

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).reshape(b, c // 2, -1)
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()

        if self.with_d_pos:
            x_g = (self.conv_g(x) + self.pos_embedding_decoder).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()
        else:
            x_g = self.conv_g(x).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)

        if self.with_pos:
            mul_theta_phi = mul_theta_phi + self.pos_embedding  # 2, 4, 32

        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().reshape(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out

class CGCNet(nn.Module):

    def __init__(self, out_channels=1):
        super(CGCNet, self).__init__()

        # resnet
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('networks/resnet34-b627a593.pth'))
        self.resnet = resnet
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # TODO add encoder
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        # TODO add decoder
        dims_decoder = [512, 256, 128, 64]
        self.decoder3 = DecoderBlock(dims_decoder[1], dims_decoder[2])
        self.decoder2 = DecoderBlock(dims_decoder[2], dims_decoder[3])
        self.decoder1 = DecoderBlock(dims_decoder[3], dims_decoder[3])
        self.finaldeconv1 = nn.ConvTranspose2d(dims_decoder[3], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, out_channels, 3, padding=1)

        # TODO 4layer
        self.conv_pred_4layer = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self.conv_pred_back_4layer = nn.Conv2d(32, 256, kernel_size=3, padding=1)

        # TODO NonLocalBlock
        self.NonLocalBlock = NonLocalBlock(channel=32, size=(32, 32))



    def NonLocal_block_4layer(self, x1):
        # 降维
        x1 = self.conv_pred_4layer(x1)
        x1 = self.NonLocalBlock(x1)
        x1 = self.conv_pred_back_4layer(x1)
        return x1


    def forward_features_linknet_4layer(self, x):

        skip_list = []

        # TODO vmlinknet init
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)


        e1_l = self.encoder1(x)
        skip_list.append(e1_l)

        e2_l = self.encoder2(e1_l)
        skip_list.append(e2_l)

        e3_l = self.encoder3(e2_l)
        skip_list.append(e3_l)

        return e3_l, skip_list


    def up_features_linknet_Center_NonLocal_4layer(self, x, skip_list):

        # TODO 交叉注意力
        x = x + self.NonLocal_block_4layer(x)

        d3 = self.decoder3(x) + skip_list[1]

        d2 = self.decoder2(d3) + skip_list[0]


        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return out



    def forward(self, x1):

        # TODO encoder
        x1, skip_list = self.forward_features_linknet_4layer(x1)

        # TODO decoder
        x = self.up_features_linknet_Center_NonLocal_4layer(x1, skip_list)

        return self.sigmoid(x)
