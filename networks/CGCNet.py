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

class CompactGlobalContextawareBlock(nn.Module):
    def __init__(self, in_channels, size=(64, 64)):
        super().__init__()

        self.in_channels = in_channels
        self.inter_channel = self.in_channels // 2
        self.conv_g = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channel, kernel_size=1, stride=1,
                                padding=0, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=self.inter_channel, out_channels=self.in_channels, kernel_size=1, stride=1,
                                   padding=0, bias=False)

        self.pooling_size = 2
        self.token_len = self.pooling_size * self.pooling_size

        self.to_qk = nn.Linear(self.in_channels, 2 * self.inter_channel, bias=False)


        self.conv_a = nn.Conv2d(in_channels, self.token_len, kernel_size=1,
                                padding=0, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(self.in_channels, self.in_channels // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.in_channels // 16, self.in_channels, bias=False),
            nn.Sigmoid()
        )

        self.with_pos = True
        if self.with_pos:
            self.pos_embedding = nn.Parameter(torch.randn(1, 4, in_channels))

        self.with_pos_2 = True
        if self.with_pos_2:
            self.pos_embedding_2 = nn.Parameter(torch.randn(1, self.inter_channel,
                                                                  size[0], size[1]))

    def compact_representation(self, x):
        b, c, h, w = x.shape
        spatial_attention = self.conv_a(x)
        spatial_attention = spatial_attention.view([b, self.token_len, -1]).contiguous()
        spatial_attention = torch.softmax(spatial_attention, dim=-1)
        x = x.view([b, c, -1]).contiguous()

        # channel attention
        channel_attention = self.avg_pool(x).view(b, c)
        channel_attention = self.fc(channel_attention).view(b, c, 1)
        x = x * channel_attention

        tokens = torch.einsum('bln,bcn->blc', spatial_attention, x)

        return tokens

    def forward(self, x):

        # [N, C, H , W]
        b, c, h, w = x.size()

        x_clone = x

        x = self.compact_representation(x)

        if self.with_pos:
            x = x + self.pos_embedding

        _, n, _ = x.size()
        qk = self.to_qk(x).chunk(2, dim=-1)
        q, k = qk[0].reshape(b, -1, n), qk[1]

        if self.with_pos_2:
            x_g = (self.conv_g(x_clone) + self.pos_embedding_2).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()
        else:
            x_g = self.conv_g(x_clone).reshape(b, c // 2, -1).permute(0, 2, 1).contiguous()

        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(q, k)

        mul_theta_phi = self.softmax(mul_theta_phi)
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(x_g, mul_theta_phi)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().reshape(b, self.inter_channel, h, w)
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g)

        out = mask + x_clone

        return out

class CGCNet(nn.Module):

    def __init__(self, out_channels=1):
        super(CGCNet, self).__init__()

        # resnet
        resnet = models.resnet34(pretrained=False)
        resnet.load_state_dict(torch.load('network/resnet34-b627a593.pth'))
        self.resnet = resnet
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # encoder
        self.first_conv = resnet.conv1
        self.first_bn = resnet.bn1
        self.first_relu = resnet.relu
        self.first_maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3

        # decoder
        dims_decoder = [512, 256, 128, 64]
        self.decoder3 = DecoderBlock(dims_decoder[1], dims_decoder[2])
        self.decoder2 = DecoderBlock(dims_decoder[2], dims_decoder[3])
        self.decoder1 = DecoderBlock(dims_decoder[3], dims_decoder[3])
        self.final_deconv1 = nn.ConvTranspose2d(dims_decoder[3], 32, 4, 2, 1)
        self.final_relu1 = nonlinearity
        self.final_conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.final_relu2 = nonlinearity
        self.final_conv3 = nn.Conv2d(32, out_channels, 3, padding=1)


        # Dimensionality Reduction
        self.reduction_conv = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        # Dimensionality Increase
        self.increase_conv = nn.Conv2d(32, 256, kernel_size=3, padding=1)

        # input shape = 1024 ——> size(64,64)  / input shape = 512 ——> size(32,32)
        self.cgcb = CompactGlobalContextawareBlock(in_channels=32, size=(32, 32))


    def compact_global_contextaware_block_reduction_increase(self, x1):
        # Dimensionality Reduction
        x1 = self.reduction_conv(x1)
        x1 = self.cgcb(x1)
        # Dimensionality Increase
        x1 = self.increase_conv(x1)
        return x1


    def forward_features(self, x):

        skip_list = []

        x = self.first_conv(x)
        x = self.first_bn(x)
        x = self.first_relu(x)
        x = self.first_maxpool(x)


        e1_l = self.encoder1(x)
        skip_list.append(e1_l)

        e2_l = self.encoder2(e1_l)
        skip_list.append(e2_l)

        e3_l = self.encoder3(e2_l)
        skip_list.append(e3_l)

        return e3_l, skip_list


    def up_features(self, x, skip_list):

        x = x + self.compact_global_contextaware_block_reduction_increase(x)

        d3 = self.decoder3(x) + skip_list[1]

        d2 = self.decoder2(d3) + skip_list[0]

        d1 = self.decoder1(d2)

        out = self.final_deconv1(d1)
        out = self.final_relu1(out)
        out = self.final_conv2(out)
        out = self.final_relu2(out)
        out = self.final_conv3(out)

        return out


    def forward(self, x1):

        # encoder
        x1, skip_list = self.forward_features(x1)

        # decoder
        x = self.up_features(x1, skip_list)

        return self.sigmoid(x)
