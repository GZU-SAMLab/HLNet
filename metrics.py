import numpy as np
import math
from skimage.metrics import structural_similarity as SSIM
from skimage.metrics import peak_signal_noise_ratio as PSNR

def compare_ergas(x_true, x_pred, ratio):
    '''
    Calculate ERGAS, ERGAS offers a global indication of the quality of fused image.The ideal value is 0.
    :param x_true:
    :param x_pred:
    :param ratio: 上采样系数
    :return:
    '''
    x_true, x_pred = img_2d_mat(x_true=x_true, x_pred=x_pred)
    sum_ergas = 0
    for i in range(x_true.shape[0]):
        vec_x = x_true[i]
        vec_y = x_pred[i]
        err = vec_x - vec_y
        r_mse = np.mean(np.power(err, 2))
        tmp = r_mse / (np.mean(vec_x)**2)
        sum_ergas += tmp
    return (100 / ratio) * np.sqrt(sum_ergas / x_true.shape[0])

def img_2d_mat(x_true, x_pred):
    '''
    # 将三维的多光谱图像转为2位矩阵
    :param x_true: (H, W, C)
    :param x_pred: (H, W, C)
    :return: a matrix which shape is (C, H * W)
    '''
    h, w, c = x_true.shape
    x_true, x_pred = x_true.astype(np.float32), x_pred.astype(np.float32)
    x_mat = np.zeros((c, h * w), dtype=np.float32)
    y_mat = np.zeros((c, h * w), dtype=np.float32)
    for i in range(c):
        x_mat[i] = x_true[:, :, i].reshape((1, -1))
        y_mat[i] = x_pred[:, :, i].reshape((1, -1))
    return x_mat, y_mat

'''
img1.shape: (512, 512, 31)
img2.shape: (512, 512, 31)
'''
def MPSNR(img1, img2, data_range):
    ch = np.size(img1,2)
    if ch == 1:
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        PIXEL_MAX = data_range
        s = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
        return s
    else:
        sum = 0
        for i in range(ch):
            mse = np.mean((img1[:,:,i] - img2[:,:,i]) ** 2)
            if mse == 0:
                return 100
            PIXEL_MAX = data_range
            s = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
            sum = sum + s
        s = sum / ch
        return s

def MPSNR1(img1, img2):
    ch = np.size(img1,2)
    if ch == 1:
        # mse = np.mean((img1 - img2) ** 2)
        # if mse == 0:
        #     return 100
        # PIXEL_MAX = 255.0
        s = PSNR(img1, img2)
        return s
    else:
        sum = 0
        for i in range(ch):
            # mse = np.mean((img1[:,:,i] - img2[:,:,i]) ** 2)
            # if mse == 0:
            #     return 100
            # PIXEL_MAX = 1.0
            s = PSNR(img1[:,:,i], img2[:,:,i])
            sum = sum + s
        s = sum / ch
        return s



def MSSIM(img1, img2, data_range):
    '''
    :param img1: shape: (512, 512, 31)
    :param img2: shape: (512, 512, 31)
    :return:
    '''
    ch = np.size(img1, 2)
    if ch == 1:
        s = SSIM(img1, img2, data_range=data_range)
        return s
    else:
        sum = 0
        for i in range(ch):
            s = SSIM(img1[:, :, i], img2[:, :, i], data_range=data_range)
            sum = sum + s
        s = sum / ch
        return s




def Loss_SAM1(x_true, x_pred):
    '''
    :param x_true: 高光谱图像：格式：(H, W, C)
    :param x_pred: 高光谱图像：格式：(H, W, C)
    :return: 计算原始高光谱数据与重构高光谱数据的光谱角相似度
    '''
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


def MSAM(img1, img2):
    '''
    :param img1: shape: (512, 512, 31)
    :param img2: shape: (512, 512, 31)
    :return:
    '''
    return Loss_SAM1(img1, img2)


def RMSE(img1, img2):

    assert img1.shape == img2.shape
    img1, img2 = img1.astype(np.float32), img2.astype(np.float32)

    return np.linalg.norm(img1 - img2) / (np.sqrt(img1.shape[0] * img1.shape[1] * img1.shape[2]))


def sum_dict(a, b):
    temp = dict()
    for key in a.keys()| b.keys():
        temp[key] = sum([d.get(key, 0) for d in (a, b)])
    return temp

def quality_assessment(x_true, x_pred, data_range, ratio, multi_dimension=False, block_size=8):
    '''
    :param multi_dimension:
    :param ratio:
    :param data_range:
    :param x_true:
    :param x_pred:
    :param block_size
    :return:
    '''
    result = {
              'MPSNR': MPSNR(x_true, x_pred, data_range=data_range),
              'MSSIM': MSSIM(x_true, x_pred, data_range=data_range),
              'ERGAS': compare_ergas(x_true=x_true, x_pred=x_pred, ratio=ratio),
              'SAM': MSAM(x_true, x_pred),
              'RMSE': RMSE(x_true, x_pred),
              }
    return result
