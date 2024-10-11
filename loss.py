import torch
import torch.nn as nn
from torch.autograd import Variable as V

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import cv2
import numpy as np


class weighted_cross_entropy(nn.Module):
    def __init__(self, num_classes=12, batch=True):
        super(weighted_cross_entropy, self).__init__()
        self.batch = batch
        self.weight = torch.Tensor([52.] * num_classes).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=self.weight)

    def __call__(self, y_true, y_pred):
        y_ce_true = y_true.squeeze(dim=1).long()

        a = self.ce_loss(y_pred, y_ce_true)

        return a


class dice_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_loss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):

        b = self.soft_dice_loss(y_true, y_pred)
        return b


def test_weight_cross_entropy():
    N = 4
    C = 12
    H, W = 128, 128

    inputs = torch.rand(N, C, H, W)
    targets = torch.LongTensor(N, H, W).random_(C)
    inputs_fl = Variable(inputs.clone(), requires_grad=True)
    targets_fl = Variable(targets.clone())
    print(weighted_cross_entropy()(targets_fl, inputs_fl))


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):

        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):

        # TODO train BiReNet or others
        loss = 0
        aux_loss_w = 1
        train_BiReNet = False
        if train_BiReNet:
            for index, pred in enumerate(y_pred):
                # 将y_true转换成道路边缘的标签开始
                if index == 1:
                    seg_label_numpy = y_true.cpu().numpy()
                    seg_label_numpy[seg_label_numpy == 255] = 1
                    kernel_size = 3
                    # 设置膨胀和腐蚀的核大小
                    kernel = np.ones((kernel_size, kernel_size), np.uint8)
                    for i in range(seg_label_numpy.shape[0]):
                        # 膨胀操作
                        dilated = cv2.dilate(seg_label_numpy[i, 0, :, :].astype(np.uint8), kernel, iterations=1)
                        # 腐蚀操作
                        eroded = cv2.erode(dilated, kernel, iterations=1)
                        # 边缘标签即为膨胀后的图像与腐蚀后的图像的差
                        seg_label_numpy[i, 0, :, :] = dilated - eroded

                    seg_label = torch.from_numpy(seg_label_numpy)
                    y_true = seg_label.to(device='cuda' if torch.cuda.is_available() else 'cpu')
                # 将y_true转换成道路边缘的标签结束
                a = self.bce_loss(pred, y_true)
                b = self.soft_dice_loss(y_true, pred)
                if index == 0:
                    loss += a + b
                else:
                    loss += aux_loss_w * (a + b)
        else:
            a = self.bce_loss(y_pred, y_true)
            b = self.soft_dice_loss(y_true, y_pred)
            loss = a + b

        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_true):
        smooth = 0.0  # may change
        i = torch.sum(y_true)
        j = torch.sum(y_pred)
        intersection = torch.sum(y_true * y_pred)

        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return 1 - score.mean()


class MulticlassDiceLoss(nn.Module):
    """
    requires one hot encoded target. Applies DiceLoss on each class iteratively.
    requires input.shape[0:1] and target.shape[0:1] to be (N, C) where N is
      batch size and C is number of classes
    """

    def __init__(self):
        super(MulticlassDiceLoss, self).__init__()

    def forward(self, input, target, weights=None):

        C = target.shape[1]

        # if weights is None:
        # 	weights = torch.ones(C) #uniform weights for all classes

        dice = DiceLoss()
        totalLoss = 0

        for i in range(C):
            diceLoss = dice(input[:, i, :, :], target[:, i, :, :])
            if weights is not None:
                diceLoss *= weights[i]
            totalLoss += diceLoss

        return totalLoss


class focal_tversky_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal = FocalLoss
        self.tversky = TverskyLoss

    def forward(self, y_true, y_pred):
        a = self.focal(y_pred, y_true)
        b = self.tversky(y_pred, y_true)

        return a + b


def FocalLoss(y_pred, y_true, alpha=0.25, gamma=2, reduction='mean'):
    ce_loss = F.cross_entropy(y_pred, y_true.squeeze(dim=1).long(), reduction='none')
    pt = torch.exp(-ce_loss)
    focal_loss = alpha * (1 - pt) ** gamma * ce_loss

    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    elif reduction == 'none':
        return focal_loss


def TverskyLoss(y_pred, y_true, alpha=0.7, beta=0.3, smooth=0):
    # Apply softmax to the predictions
    y_pred = F.softmax(y_pred, dim=1)

    # Calculate the number of channels
    num_channels = y_pred.size(1)

    total_loss = 0
    for channel in range(num_channels):
        pre_channel = y_pred[:, channel, :, :]
        true_channel = (y_true == channel).float()

        TP = torch.sum(pre_channel * true_channel)
        FN = torch.sum((1 - pre_channel) * true_channel)
        FP = torch.sum(pre_channel * (1 - true_channel))

        score = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        total_loss += 1 - score
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
    return total_loss / num_channels
