# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
from torch.optim.lr_scheduler import CosineAnnealingLR
import random
import warnings
import datetime
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
import joblib
from skimage.io import imread

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

import loss
from dataset.dataset import Dataset
from net.SNAU_Net_run import SNAU_Net
from net.Swin_run_lcab import lcab_two


from utilities.metrics import dice_coef, batch_iou, mean_iou, iou_score
import utilities.losses as losses
from utilities.utils import str2bool, count_params
import pandas as pd
from net import Unet,res_unet_plus,R2Unet,sepnet,ResU_Net, Net_dilation

from config import get_config
from net.swinunet import SwinTransformerSys

arch_names = list(Unet.__dict__.keys())
loss_names = list(losses.__dict__.keys())
loss_names.append('BCEWithLogitsLoss')


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--deepsupervision', default=False,
                        help='model name: (default: arch+timestamp)')

    parser.add_argument('--arch', '-a', metavar='ARCH', default='SNAU_NET_TWO',
                        # choices=arch_names,
                        help='model architecture: ' +
                            ' | '.join(arch_names) +
                            ' (default: NestedUNet)')

    parser.add_argument('--dataset', default="Lits_tumor",
                        help='dataset name')
    parser.add_argument('--input-channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--image-ext', default='npy',
                        help='image file extension')
    parser.add_argument('--mask-ext', default='npy',
                        help='mask file extension')
    parser.add_argument('--aug', default=False, type=str2bool)
    parser.add_argument('--loss', default='BCELogitsLoss',
                        choices=loss_names,
                        help='loss: ' +
                            ' | '.join(loss_names) +
                            ' (default: BCEDiceLoss)')

    parser.add_argument('--epochs', default=250, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--early-stop', default=80, type=int,
                        metavar='N', help='early stopping (default: 30)')

    parser.add_argument('-b', '--batch-size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                            ' | '.join(['Adam', 'SGD']) +
                            ' (default: Adam)')
    parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    parser.add_argument('--seed', type=int,
                        default=7, help='random seed')
    args = parser.parse_args()

    return args

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum =self.sum+val * n
        self.count=self.count + n
        self.avg = self.sum / self.count

def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def train(args, train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()
    torch.autograd.set_detect_anomaly(True)
    model.train()

    for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        input = input.cuda()
        target = target.cuda()

        # compute output
        if args.deepsupervision:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss = loss.clone()+criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou= iou_score(output, target)
            dice_1, dice_2 = dice_coef(output, target)

        losses.update(loss.item(), input.size(0))
        ious.update(iou, input.size(0))
        dices_1s.update(torch.tensor(dice_1), input.size(0))
        dices_2s.update(torch.tensor(dice_2), input.size(0))

        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])

    return log


def validate(args, val_loader, model, criterion):
    losses = AverageMeter()
    ious = AverageMeter()
    dices_1s = AverageMeter()
    dices_2s = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if args.deepsupervision:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss = loss.clone()+criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice_1 = dice_coef(output, target)[0]
                dice_2 = dice_coef(output, target)[1]

            losses.update(loss.item(), input.size(0))
            ious.update(iou, input.size(0))
            dices_1s.update(torch.tensor(dice_1), input.size(0))
            dices_2s.update(torch.tensor(dice_2), input.size(0))

    log = OrderedDict([
        ('loss', losses.avg),
        ('iou', ious.avg),
        ('dice_1', dices_1s.avg),
        ('dice_2', dices_2s.avg)
    ])

    return log


def main():
    args = parse_args()
    set_random_seed(args.seed, deterministic=True)

    if args.name is None:
        if args.deepsupervision:
            args.name = '%s_%s_lym' %(args.dataset, args.arch)
        else:
            args.name = '%s_%s_lym' %(args.dataset, args.arch)
    timestamp  = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    if not os.path.exists('models/{}/{}'.format(args.name,timestamp)):
        os.makedirs('models/{}/{}'.format(args.name,timestamp))

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    with open('models/{}/{}/args.txt'.format(args.name,timestamp), 'w') as f:
        for arg in vars(args):
            print('%s: %s' %(arg, getattr(args, arg)), file=f)

    joblib.dump(args, 'models/{}/{}/args.pkl'.format(args.name,timestamp))

    # define loss function (criterion)
    if args.loss == 'BCEWithLogitsLoss':
        criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.BCEDiceLoss().cuda()

    cudnn.benchmark = True

    # # Data loading code
    train_img_paths = glob('./data/tumor/trainImage/*')
    train_mask_paths = glob('./data/tumor/trainMask/*')
    #
    val_img_paths = glob('./data/tumor/validImage/*')
    val_mask_paths = glob('./data/tumor/validMask/*')

    print("train_num:%s"%str(len(train_img_paths)))
    print("val_num:%s"%str(len(val_img_paths)))

    # create model

    config = get_config(args)
    model = lcab_two(config, img_size=224, num_classes=2)

    model.load_from(config)
    model = torch.nn.DataParallel(model).cuda()


    # model = torch.nn.DataParallel(model).cuda()

    print(count_params(model))

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
            momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)


    train_dataset = Dataset(args, train_img_paths, train_mask_paths, transform=True)
    val_dataset = Dataset(args, val_img_paths, val_mask_paths, transform=False)


    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    log = pd.DataFrame(index=[], columns=[
        'epoch', 'lr', 'loss', 'iou','dice_1', 'dice_2', 'val_loss', 'val_iou','val_dice_1', 'val_dice_2'
    ])

    best_loss = 100
    best_dice2 = 0
    best_iou = 0
    trigger = 0
    first_time = time.time()
    for epoch in range(args.epochs):
        print('Epoch [%d/%d]' %(epoch, args.epochs))
        # train for one epoch
        train_log = train(args, train_loader, model, criterion, optimizer, epoch)

        val_log = validate(args, val_loader, model, criterion)
        # scheduler.step()

        print('loss %.4f - iou %.4f - dice_1 %.4f - dice_2 %.4f - val_loss %.4f - val_iou %.4f - val_dice_1 %.4f - val_dice_2 %.4f'
                  %(train_log['loss'], train_log['iou'], train_log['dice_1'], train_log['dice_2'], val_log['loss'], val_log['iou'], val_log['dice_1'], val_log['dice_2']))

        # print('loss %.4f - iou %.4f - dice %.4f ' %(train_log['loss'], train_log['iou'], train_log['dice']))
        end_time = time.time()
        print("time:", (end_time - first_time) / 60)

        tmp = pd.Series([
            epoch,
            args.lr,
            train_log['loss'],
            train_log['iou'],
            train_log['dice_1'],
            train_log['dice_2'],
            val_log['loss'],
            val_log['iou'],
            val_log['dice_1'],
            val_log['dice_2'],
        ], index=['epoch', 'lr', 'loss', 'iou', 'dice_1' ,'dice_2' ,'val_loss', 'val_iou', 'val_dice_1' ,'val_dice_2'])

        log = log.append(tmp, ignore_index=True)
        log.to_csv('models/{}/{}/log.csv'.format(args.name,timestamp), index=False)

        trigger += 1

        val_loss = val_log['loss']
        #
        torch.save(model.state_dict(), 'models/{}/{}/epoch{}-{:.4f}-{:.4f}_model.pth'.format(args.name,timestamp,epoch,val_log['dice_1'],val_log['dice_2']))


        val_dice = val_log['dice_2']
        # if val_loss < best_loss:
        if val_dice > best_dice2:
            torch.save(model.state_dict(),
                       'models/{}/{}/epoch{}-{:.4f}-{:.4f}_model.pth'.format(args.name, timestamp, epoch, val_log['dice_1'], val_log['dice_2']))

            best_dice2 = val_dice
            print("=> saved best model")
            trigger = 0

    torch.cuda.empty_cache()

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    main()