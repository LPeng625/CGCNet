# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import cv2
import os
from time import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V

# networks中定义了模型。导入模型
# from networks.cenet import CE_Net_
# 导入自己的模型
# from networks.our_net import CE_attention_Net_
from networks.BiReNet import CGCNet
from networks.RCFSNet import RCFSNet
from networks.DLinkNet import DinkNet34
from networks.Nlinknet import NLinkNet34
from networks.deeplabv3plus import DeepLabv3_plus
from networks.DBRANet import DBRANet
from networks.MACUNet import MACUNet
from networks.Linknet import LinkNet34
# framework：保存基本框架，测试图像，进行训练，导入保存的模型等功能的函数
from framework import MyFrame
# loss：定义损失函数，diceloss， 多类的diceloss，focalloss等
from loss import dice_bce_loss, focal_tversky_loss
# data ：定义数据，定义数据的导入，数据的扩增函数
from data import ImageFolder
# 可视化工具
# from Visualizer import Visualizer

# 定义超参数，图像大小，图像的
import Constants

###########修改使用的显卡
# Please specify the ID of graphics cards that you want to use
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def CE_Net_Train():
    NAME = 'D-CGCNet' + Constants.ROOT.split('/')[-1]

    print(NAME)
    # run the Visdom
    # viz = Visualizer(env=NAME)

    solver = MyFrame(CGCNet, dice_bce_loss, 2e-4)
    # solver.load('./weights/CGCNet-FFM_CSAM_onlyMP_2.th')
    batchsize = torch.cuda.device_count() * Constants.BATCHSIZE_PER_CARD
    # For different 2D medical image segmentation tasks, please specify the dataset which you use
    # for examples: you could specify "dataset = 'DRIVE' " for retinal vessel detection.

    #############修改自己的数据集地址
    dataset = ImageFolder(root_path=Constants.ROOT, datasets='DRIVE')

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batchsize,
        shuffle=True,
        num_workers=0)

    # start the logging files
    mylog = open('logs/' + NAME + '.log', 'w')
    tic = time()

    no_optim = 0
    total_epoch = Constants.TOTAL_EPOCH
    train_epoch_best_loss = Constants.INITAL_EPOCH_LOSS
    iters_per_epoch = len(dataset) // batchsize
    for epoch in range(1, total_epoch + 1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        index = 0

        pbar = tqdm(enumerate(data_loader), total=iters_per_epoch,
                    desc=f"Epoch: [{epoch + 1}/{total_epoch + 1}] Iter: [{0}/{iters_per_epoch}] LR: {solver.old_lr:.8f} Loss: {train_epoch_loss:.8f}")

        for iters, (img, mask) in pbar:
            # for img, mask in data_loader_iter:
            solver.set_input(img, mask)
            train_loss, pred = solver.optimize()
            train_epoch_loss += train_loss
            index = index + 1

            pbar.set_description(
                f"Epoch: [{epoch + 1}/{total_epoch + 1}] Iter: [{iters + 1}/{iters_per_epoch}] LR: {solver.old_lr:.8f} Loss: {train_epoch_loss / (iters + 1):.8f}  Loss.item: {train_loss.item():.8f}")

        # show the original images, predication and ground truth on the visdom.

        # ########归一化的方法为什么？？？
        # show_image = (img + 1.6) / 3.2 * 255.
        # viz.img(name='images', img_=show_image[0, :, :, :])
        # viz.img(name='labels', img_=mask[0, :, :, :])
        # viz.img(name='prediction', img_=pred[0, :, :, :])

        train_epoch_loss = train_epoch_loss / len(data_loader_iter)
        # 将信息保存在log文件夹中
        print('********', file=mylog)
        print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
        print('train_loss:', train_epoch_loss, file=mylog)
        print('SHAPE:', Constants.Image_size, file=mylog)
        print('********')
        print('epoch:', epoch, '    time:', int(time() - tic))
        print('train_loss:', train_epoch_loss)
        print('SHAPE:', Constants.Image_size)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('./weights/' + NAME + '.th')

        if no_optim > Constants.NUM_EARLY_STOP:
            print('early stop at %d epoch' % epoch, file=mylog)
            print('early stop at %d epoch' % epoch)
            break
        if no_optim > Constants.NUM_UPDATE_LR:
            if solver.old_lr < 5e-7:
                break
            solver.load('./weights/' + NAME + '.th')
            solver.update_lr(5.0, factor=True, mylog=mylog)
        mylog.flush()

    print('Finish!', file=mylog)
    print('Finish!')
    mylog.close()


if __name__ == '__main__':
    print(torch.__version__)
    CE_Net_Train()
