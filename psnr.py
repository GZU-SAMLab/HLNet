import numpy as np
import math
from skimage.metrics import structural_similarity as SSIM
import torch

def MPSNR(img1, img2):
    ch = np.size(img1,2)
    if ch == 1:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = 1.0
        s = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        return s
    else:
        sum = 0
        for i in range(ch):
            mse = np.mean((img1[:,:,i] - img2[:,:,i]) ** 2)
            if mse == 0:
                return 100
            PIXEL_MAX = 1.0
            s = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
            sum = sum + s
        s = sum / ch
        return s

def MSSIM(img1, img2):
    ch = np.size(img1,2)
    if ch == 1:
        s = SSIM(img1, img2, data_range=1.0)
        return s
    else:
        sum = 0
        for i in range(ch):
            s = SSIM(img1[:,:,i], img2[:,:,i], data_range=1.0)
            sum = sum + s
        s = sum / ch
        return s

def Loss_SAM(im_fake, im_true):
    sum1 = torch.sum(im_true * im_fake, 1)
    sum2 = torch.sum(im_true * im_true, 1)
    sum3 = torch.sum(im_fake * im_fake, 1)
    t = (sum2 * sum3) ** 0.5
    numlocal = torch.gt(t, 0)
    num = torch.sum(numlocal)
    t = sum1 / t
    angle = torch.acos(t)
    sumangle = torch.where(torch.isnan(angle), torch.full_like(angle, 0), angle).sum()
    if num == 0:
        averangle = sumangle
    else:
        averangle = sumangle / num
    SAM = averangle * 180 / 3.14159256
    return SAM

def SAM1(x_true, x_pred):
    num = 0
    sum_sam = 0
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    for x in range(x_true.shape[0]):
        for y in range(x_true.shape[1]):
            tmp_pred = x_pred[x, y].ravel()
            tmp_true = x_true[x, y].ravel()
            if np.linalg.norm(tmp_true) != 0 and np.linalg.norm(tmp_pred) != 0:
                sum_sam += np.arccos(
                    np.minimum(1, np.inner(tmp_pred, tmp_true) / (np.linalg.norm(tmp_true) * np.linalg.norm(tmp_pred))))

                num += 1
    sam_deg = (sum_sam / num) * 180 / np.pi
    return sam_deg

'''
type:torch.tensor
shape: b c h w
img1.shape: (512, 512, 31)
img2.shape: (512, 512, 31)
'''
def MSAM(img1, img2):
    return Loss_SAM(img1, img2)
    
    
def RMSE(img1, img2):
    assert img1.shape == img2.shape
    error = img1 - img2
    sqrt_error = torch.pow(error, 2)
    rmse = torch.sqrt(torch.mean(sqrt_error.contiguous().view(-1)))    
    return rmse
