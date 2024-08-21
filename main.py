from __future__ import print_function
import argparse
from math import log10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from m6model import SpatialSpectralSRNet_test
from data import get_patch_training_set, get_test_set
from torch.nn import init
import skimage.measure
from torch.autograd import Variable
from psnr import MPSNR,MSSIM,MSAM,RMSE,SAM1
import numpy as np
import math
import scipy.io as io
import os
import random
from torch.utils.tensorboard import SummaryWriter
from metrics import quality_assessment, sum_dict

from thop import profile


# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=2, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--ChDim', type=int, default=102, help='output channel number')
parser.add_argument('--alpha', type=float, default=0.4, help='alpha')
parser.add_argument('--nEpochs', type=int, default=0, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0008, help='Learning Rate. Default=0.01')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--save_folder', default='TrainedNet/', help='Directory to keep training outputs.')
parser.add_argument('--outputpath', type=str, default='result/', help='Path to output img')
parser.add_argument('--n_dlb', type=int, default=1, help='Num of DLB')
parser.add_argument('--mode', default=1, type=int, help='Train or Test.')
parser.add_argument('--beta', default=0.1, type=float, help='beta of li.')

opt = parser.parse_args()

print(opt)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(opt.seed)

print('===> Loading datasets')
if opt.mode == 1:
    train_set = get_patch_training_set(opt.upscale_factor, opt.patch_size)
    print('===> Training_data loading complete') 
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True,
                                      pin_memory=True)

test_set = get_test_set(opt.upscale_factor)
print('===> Testing_data loading complete') 
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False,
                                 pin_memory=True)

print('===> Building model')
writer = SummaryWriter()

def mkdir(path):
    folder = os.path.exists(path)

    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")

mkdir(opt.save_folder)
mkdir(opt.outputpath)

model = SpatialSpectralSRNet_test(out_channels = opt.ChDim, n_dlb = opt.n_dlb, upscale_factor = opt.upscale_factor).cuda()
print('# network parameters: {}'.format(sum(param.numel() for param in model.parameters())))
fake_input = torch.randn(1, 4, 64, 64).cuda()
flops, params=profile(model, (fake_input,))
# print('flops: ', flops, 'params: ', params)
# print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))



optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=[25, 50, 100, 200], gamma=0.5)

if opt.nEpochs != 0:
    load_dict = torch.load(opt.save_folder + "_epoch_{}.pth".format(opt.nEpochs))
    opt.lr = load_dict['lr']
    epoch = load_dict['epoch']
    model.load_state_dict(load_dict['param'])
    optimizer.load_state_dict(load_dict['adam'])

criterion = nn.L1Loss()

def interval_loss(HX, X, inval):

    HX_dist = HX[:, inval[1:], :, :] - HX[:, inval[:-1], :, :]
    X_dist = X[:, inval[1:], :, :] - X[:, inval[:-1], :, :]

    return abs(HX_dist), abs(X_dist)


def train(epoch, optimizer):
    epoch_loss = 0
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        with torch.autograd.set_detect_anomaly(True):
            Y, X_1, X_2, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda(), batch[3].cuda()
            optimizer.zero_grad()
            Y = Variable(Y).float() # LR-MSI
            X_1 = Variable(X_1).float() # HR-MSI
            X_2 = Variable(X_2).float() # LR-HSI
            X = Variable(X).float() # HR-HSI
            spa_X, spe_X, HX = model(Y)
            
            
            HX_dist, X_dist = interval_loss(HX, X, [12, 30, 58, 91])

            
            loss = criterion(HX, X) + opt.alpha*criterion(spe_X, X_2) + opt.alpha*criterion(spa_X, X_1) + opt.beta*criterion(HX_dist, X_dist)
            # loss = criterion(HX, X)  + opt.beta*criterion(HX_dist, X_dist)
            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()


            if iteration%100==0:
                print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader)


def test():
    avg_psnr = 0
    avg_ssim = 0
    avg_sam = 0
    avg_sam1 = 0
    avg_rmse = 0
    indices = dict()
    with torch.no_grad():
        for batch in testing_data_loader:
            Y, X = batch[0].cuda(), batch[1].cuda()
            Y = Variable(Y).float()
            X = Variable(X).float()
            spa_X, spe_X, HX = model(Y)
            X1 = torch.squeeze(X).permute(1, 2, 0).cpu()
            HX1 = torch.squeeze(HX).permute(1, 2, 0).cpu()
            X = torch.squeeze(X).permute(1, 2, 0).cpu().numpy()
            HX = torch.squeeze(HX).permute(1, 2, 0).cpu().numpy()
            spa_X = torch.squeeze(spa_X).permute(1, 2, 0).cpu().numpy()
            spe_X = torch.squeeze(spe_X).permute(1, 2, 0).cpu().numpy()
            psnr = MPSNR(HX, X)
            ssim = MSSIM(HX, X)
            sam = MSAM(HX1, X1)
            sam1 = SAM1(HX, X)
            rmse = RMSE(HX1, X1)
            im_name = batch[2][0]
            print(im_name)
            (path, filename) = os.path.split(im_name)
            io.savemat(opt.outputpath + filename, {'HX': HX})
            io.savemat('spa/' + filename, {'HX': spa_X})
            io.savemat('spe/' + filename, {'HX': spe_X})
            avg_psnr += psnr
            avg_ssim += ssim
            avg_sam += sam
            avg_sam1 += sam1
            avg_rmse += rmse
            # print(quality_assessment(HX, X, 1.0, opt.upscale_factor))
               
            if batch == 0:
                indices = quality_assessment(HX, X, 1.0, opt.upscale_factor)
            else:
                indices = sum_dict(indices, quality_assessment(HX, X, 1.0, opt.upscale_factor))
        # print(indices)
        for index in indices:
            indices[index] = indices[index] / len(testing_data_loader)
        print(indices)
        
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("===> Avg. SSIM: {:.4f} ".format(avg_ssim / len(testing_data_loader)))
    print("===> Avg. SAM: {:.4f} ".format(avg_sam / len(testing_data_loader)))
    print("===> Avg. SAM1: {:.4f} ".format(avg_sam1 / len(testing_data_loader)))
    print("===> Avg. RMSE: {:.4f} ".format(avg_rmse / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader)


def checkpoint(epoch):

    model_out_path = opt.save_folder + "_epoch_{}.pth".format(epoch)
    if epoch % 5 == 0:
        save_dict = dict(
            lr=optimizer.state_dict()['param_groups'][0]['lr'],
            param=model.state_dict(),
            adam=optimizer.state_dict(),
            epoch=epoch
        )
        torch.save(save_dict, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))

T =0
if opt.mode == 1:
    for epoch in range(opt.nEpochs + 1, 301):
        avg_loss = train(epoch, optimizer)
        checkpoint(epoch)

        # test()
        R = test()
        if R > T:
            T = R
            model_out_path = opt.save_folder + "_epoch_{}.pth".format(epoch)
            save_dict = dict(
                lr=optimizer.state_dict()['param_groups'][0]['lr'],
                param=model.state_dict(),
                adam=optimizer.state_dict(),
                epoch=epoch
            )
            torch.save(save_dict, model_out_path)

            print("Checkpoint saved to {}".format(model_out_path))
        # test()

        scheduler.step()
        print("Current best psnr result {}".format(T))


    
else:
    test()
