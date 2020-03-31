import tensorflow as tf
import numpy as np
import scipy.ndimage
import scipy
import random
import glob
import math
import os
import re

from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d


# ================================================== #
#           Statistic info of the network            #
# ================================================== #
def count_number_trainable_params():
    '''
    Counts the number of trainable variables.
    '''
    total_nb_params = 0
    total_btypes = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        variable_type = trainable_variable.dtype
        total_nb_params = total_nb_params + current_nb_params
        total_btypes = total_btypes + current_nb_params * params_bytes(variable_type)
    
    return "Model size: {0}K, Space usage: {1}KB ({2:6.2f}MB)".format(total_nb_params/1000,
                                                                      total_btypes/1000,
                                                                      total_btypes/1000000.0)

def params_bytes(vtype):
    if vtype == 'float32_ref':
        return 32 / 8
    if vtype == 'float64_ref':
        return 64 / 8

def get_nb_params_shape(shape):
    '''
    Computes the total number of params for a given shap.
    Works for any number of shapes etc [D,F] or [W,H,C] computes D*F and W*H*C.
    '''
    nb_params = 1
    for dim in shape:
        nb_params = nb_params*int(dim)
    return nb_params

# ================================================== #
#               Downsampling methods                 #
# ================================================== #
def get_gauss_filter(shape=(7, 7), sigma=1.2):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    
    if sumh != 0:
        h /= sumh
    return h


def blur(hrlf, psf):
    '''
        HR_LF = [batchsize, height, width, sview, tview, channel]
        
        return blurred_lfimgs
    '''
    blurred_lfimgs = np.zeros_like(hrlf)
    ws = psf.shape[0]
    t = (ws-1) / 2
    
    hrlf = np.concatenate([hrlf[:, :t, :], hrlf, hrlf[:, -t:, :]], axis=1)
    hrlf = np.concatenate([hrlf[:t, :, :], hrlf, hrlf[-t:, :, :]], axis=0)
    
    if hrlf.shape[2] == 3:
        blurred_lfimgs[:, :, 0] = convolve2d(hrlf[:, :, 0], psf, 'valid')
        blurred_lfimgs[:, :, 1] = convolve2d(hrlf[:, :, 1], psf, 'valid')
        blurred_lfimgs[:, :, 2] = convolve2d(hrlf[:, :, 2], psf, 'valid')
    else:
        blurred_lfimgs = convolve2d(np.squeeze(hrlf), psf, 'valid')
        blurred_lfimgs = np.expand_dims(blurred_lfimgs, axis=2)
    
    return blurred_lfimgs

def downsampling(HR_LF, K1=2, K2=2, nSig=1.2, spatial_only=False):
    '''HR_LF = [batchsize, height, width, sview, tview, channels]
    '''
    b, h, w, s, t, c = HR_LF.shape
    psf = get_gauss_filter(shape=(7, 7), sigma=nSig)
    LR_LF = np.zeros_like(HR_LF)
    for n in range(b):
        for i in range(s):
            for j in range(t):
                LR_LF[n, :, :, i, j, :] = blur(HR_LF[n, :, :, i, j, :], psf)
    
    LR_LF = LR_LF[:, ::K1, ::K1, :, :, :]
    if not spatial_only:
        LR_LF = LR_LF[:, :, :, ::K2, ::K2, :]
    
    return LR_LF

# ================================================== #
#                   Metric Methods                   #
# ================================================== #
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def lfpsnrs(truth4d, recons4d):
    '''The truth4d represents for a single 4d-patches of light field
        
        truth4d  = [height, width, sview, tview, channels]
        recons4d = [height, width, sview, tview, channels]
    '''
    assert truth4d.shape == recons4d.shape, 'The prediction and label should be same size.'
    assert truth4d.dtype == 'uint8', 'The ground truth should be uint8 format within range (0, 255)'
    assert recons4d.dtype == 'uint8', 'The inputs should be uint8 format within range (0, 255)'

    h, w, s, t = np.squeeze(truth4d).shape
    lfpsnr = np.zeros([s, t])
    for i in range(s):
        for j in range(t):
            truth = truth4d[:, :, i, j]
            truth = np.squeeze(truth)
            recons = recons4d[:, :, i, j]
            recons = np.squeeze(recons)
            lfpsnr[i, j] = psnr(truth, recons)
    meanpsnr = np.mean(lfpsnr)
    return lfpsnr, meanpsnr


def batchmeanpsnr(truth, pred):
    '''This function calculate the average psnr over a batchsize of light field patches and
        the size of each batch equal to the length of inputs
        The inputs should be uint8 format within range (0, 1)
    '''
    batchmean_psnr = 0
    for i in range(len(pred)):
        _, meanpsnr = lfpsnrs(np.uint8(truth[i]*255.), np.uint8(pred[i]*255.))
        batchmean_psnr += meanpsnr
    batchmean_psnr = batchmean_psnr / len(pred)
    
    return batchmean_psnr
    
def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    mu1 = gaussian_filter(img1, sd)
    mu2 = gaussian_filter(img2, sd)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = gaussian_filter(img1 * img1, sd) - mu1_sq
    sigma2_sq = gaussian_filter(img2 * img2, sd) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sd) - mu1_mu2

    ssim_num = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))

    ssim_den = ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    ssim_map = ssim_num / ssim_den
    return np.mean(ssim_map)


def lfssims(truth4d, recons4d):
    ''' The truth4d represents for a single 4d-patches of light field
        
        truth4d  = [height, width, sview, tview, channels]
        recons4d = [height, width, sview, tview, channels]
    '''
    assert truth4d.shape == recons4d.shape, 'The prediction and label should be same size.'
    
    h, w, s, t, c = truth4d.shape
    lfssim = np.zeros([s, t])
    for i in range(s):
        for j in range(t):
            truth = truth4d[:, :, i, j, :]
            truth = np.squeeze(truth)
            recons = recons4d[:, :, i, j, :]
            recons = np.squeeze(recons)
            lfssim[i, j] = ssim_exact(truth/255., recons/255.)
    meanssim = np.mean(lfssim)
    return lfssim, meanssim

def batchmeanssim(truth, pred):
    ''' This function calculate the average psnr over a batchsize of light field patches and
        the size of each batch equal to the length of inputs
        The inputs should be uint8 format within range (0, 1)
    '''
    batchmean_ssim = 0
    for i in range(len(pred)):
        _, meanssim = lfssims(np.uint8(truth[i]*255.), np.uint8(pred[i]*255.))
        batchmean_ssim += meanssim
    batchmean_ssim = batchmean_ssim / len(pred)
    
    return batchmean_ssim