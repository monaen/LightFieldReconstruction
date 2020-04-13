import tensorflow as tf
import numpy as np
import math

from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d


# ================================================== #
#                Statistical Functions               #
# ================================================== #

def count_number_trainable_params():
    """
    Counts the number of trainable variables.

    :return: The trainable parameters of a network.
    """

    def params_bytes(vtype):
        if vtype == "float32_ref":
            return 32 / 8
        if vtype == "float64_ref":
            return 64 / 8

    def get_nb_params_shape(shape):
        nb_params = 1
        for dim in shape:
            nb_params = nb_params * int(dim)
        return nb_params

    total_nb_params = 0
    total_btypes = 0
    for trainable_variable in tf.trainable_variables():
        shape = trainable_variable.get_shape()  # e.g [D,F] or [W,H,C]
        current_nb_params = get_nb_params_shape(shape)
        variable_type = trainable_variable.dtype
        total_nb_params = total_nb_params + current_nb_params
        total_btypes = total_btypes + current_nb_params * params_bytes(variable_type)
    
    info = "Model size: {0}K, Space usage: {1}KB ({2:6.2f}MB)".format(total_nb_params/1000,
                                                                      total_btypes/1000,
                                                                      total_btypes/1000000.0)
    print(info)
    return


# ================================================== #
#               Downsampling Methods                 #
# ================================================== #

def get_gauss_filter(shape=(7, 7), sigma=1.2):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma]).

    :param shape: window size
    :param sigma: variance

    :return:      gaussian filter
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
    """
    The blur method for light field imaging which generates
    the low-quality light field.

    :param hrlf: high-resolution light field
    :param psf:  blur kernel

    :return:     blurred light field
    """
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


def downsampling(data, rs=1, ra=1, nSig=1.2):
    '''
    Dowmsampling method on spatial or angular dimensions

    :param data: high-resolution light field (labels)
    :param rs:   spatial downsampling rate
    :param ra:   angular downsampling rate
    :param nSig: gaussian noise variation

    :return:     downsampled light field
    '''
    def spatial_downsampling(GT, rate=2):
        b, h, w, s, t, c = GT.shape
        psf = get_gauss_filter(shape=(7, 7), sigma=nSig)
        downsampled = np.zeros_like(GT)
        for n in range(b):
            for i in range(s):
                for j in range(t):
                    downsampled[n, :, :, i, j, :] = blur(GT[n, :, :, i, j, :], psf)
        downsampled = downsampled[:, ::rate, ::rate, :, :, :]
        return downsampled

    def angular_downsampling(GT, rate=2):
        downsampled = None
        if rate == 4:  # 2x2 -> 8x8
            GT = GT[:, :, :, :-1, :-1, :]
            downsampled = GT[:, :, :, 0:8:7, 0:8:7, :]
        elif rate == 3:  # 3x3 -> 9x9
            downsampled = GT[:, :, :, ::(rate+1), ::(rate+1), :]
        elif rate == 2:  # 5x5 -> 9x9
            downsampled = GT[:, :, :, ::rate, ::rate, :]
        elif rate == 0:  # 3x3 -> 7x7
            GT = GT[:, :, :, 1:-1, 1:-1, :]
            downsampled = GT[:, :, :, ::3, ::3, :]
        else:
            assert False, "Unsupported angular downsampling rate: {}.".format(rate)
        return GT, downsampled

    if rs != 1 and ra == 1:
        downsampled = spatial_downsampling(data, rate=rs)
        return downsampled
    elif ra != 1 and rs == 1:
        label, downsampled = angular_downsampling(data, rate=ra)
        return label, downsampled
    elif ra != 1 and rs != 1:
        label, downsampled = angular_downsampling(data, rate=ra)
        downsampled = spatial_downsampling(downsampled, rate=rs)

        return label, downsampled
    else:
        assert False, "Both spatial and angular downsampling rates are 1."


# ================================================== #
#                   Metric Functions                 #
# ================================================== #

def psnr(img1, img2):
    """
    Metric function to calculate the PSNR value for single image.

    :param img1: input image 1
    :param img2: input image 2

    :return:     PSNR value
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def lfpsnrs(truth4d, recons4d):
    """
    Metric function to calculate the average PSNR value for a
    light field.

    :param truth4d:  ground truth light field
    :param recons4d: reconstructed light field

    :return:         average PSNR value for a light field
    """
    assert truth4d.shape == recons4d.shape, "The prediction and label should be same size."
    assert truth4d.dtype == "uint8", "The ground truth should be uint8 format within range (0, 255)"
    assert recons4d.dtype == "uint8", "The inputs should be uint8 format within range (0, 255)"

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
    """
    Metric function to calculate the average PSNR value for a batch of
    light fields.

    :param truth: a batch of ground truth light field
    :param pred:  a batch of reconstructed light field
    :return:      average PSNR value for a batch of light field
    """
    batchmean_psnr = 0
    for i in range(len(pred)):
        _, meanpsnr = lfpsnrs(np.uint8(truth[i]*255.), np.uint8(pred[i]*255.))
        batchmean_psnr += meanpsnr
    batchmean_psnr = batchmean_psnr / len(pred)
    
    return batchmean_psnr


def ssim_exact(img1, img2, sd=1.5, C1=0.01**2, C2=0.03**2):
    """
    Python version of Matlab SSIM function to calculate the SSIM value of single image.

    :param img1: input image 1
    :param img2: input image 2
    :param sd:   standard
    :param C1:   parameter C1
    :param C2:   parameter C2

    :return:     SSIM value
    """
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
    """
    Metric function to calculate the average SSIM value for a
    light field.

    :param truth4d:  ground truth light field
    :param recons4d: reconstructed light field

    :return:         average SSIM value for a light field
    """
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
    """
    Metric function to calculate the average SSIM value for a batch of
    light fields.

    :param truth: a batch of ground truth light field
    :param pred:  a batch of reconstructed light field

    :return:      average SSIM value for a batch of light field
    """
    batchmean_ssim = 0
    for i in range(len(pred)):
        _, meanssim = lfssims(np.uint8(truth[i]*255.), np.uint8(pred[i]*255.))
        batchmean_ssim += meanssim
    batchmean_ssim = batchmean_ssim / len(pred)
    
    return batchmean_ssim


# ================================================== #
#        Split patches using overlap strides         #
# ================================================== #

def shaveLF(inLF, border=(3, 3)):
    """
    Shave the input light field in terms of a given border.

    :param inLF:   input light field of size: [H, W, S, T, C]
    :param border: border values

    :return:       shaved light field
    """
    h_border, w_border = border
    if (h_border != 0) and (w_border != 0):
        shavedLF = inLF[h_border:-h_border, w_border:-w_border, ...]
    elif (h_border != 0) and (w_border == 0):
        shavedLF = inLF[h_border:-h_border, :, ...]
    elif (h_border == 0) and (w_border != 0):
        shavedLF = inLF[:, w_border:-w_border, ...]
    else:
        shavedLF = inLF
    return shavedLF


def shaveLF_by_factor(inLF, factor=4):
    """
    Shave the input light field in terms of a given upsampling factor.

    :param inLF:   input light field of size: [H, W, S, T, C]
    :param border: border values

    :return:       shaved light field
    """
    h, w, s, t, c = inLF.shape
    h_shaved = h - np.mod(h, factor)
    w_shaved = w - np.mod(w, factor)
    shavedLF = inLF[0:h_shaved, 0:w_shaved, :, :, :]
    return shavedLF


def shave_batch_LFs(inLFs, border=(3, 3)):
    """
    Shave the input light field by a given border.

    :param inLFs:  a batch of input light fields of
                   size: [B, H, W, S, T, C]
    :param border: border values

    :return:       shaved light field
    """
    h_border, w_border = border
    if (h_border != 0) and (w_border != 0):
        shavedLFs = inLFs[:, h_border:-h_border, w_border:-w_border, :, :, :]
    elif (h_border != 0) and (w_border == 0):
        shavedLFs = inLFs[:, h_border:-h_border, :, :, :, :]
    elif (h_border == 0) and (w_border != 0):
        shavedLFs = inLFs[:, :, w_border:-w_border, :, :, :]
    else:
        shavedLFs = inLFs[:, :, :, :, :, :]
    return shavedLFs


def LF_split_patches(inLF, patchsize=96, stride=30):
    """
    Split the entire light filed into patches.

    :param inLF:      input light field
    :param patchsize: size of each cropped patch
    :param stride:    stride between adjacent cropped regions

    :return:          cropped patches
    """
    _, height, width, sSize, tSize, channels = inLF.shape
    
    patch_squence = []
    indices_squence = []
    for y in range(0, height-patchsize+1, stride):
        for x in range(0, width-patchsize+1, stride):
            patch = inLF[:, y:y+patchsize, x:x+patchsize, :, :, :]
            indices = np.array([y, y+patchsize, x, x+patchsize])
            patch_squence.append(patch)
            indices_squence.append(indices)
        patch = inLF[:, y:y+patchsize, width-patchsize:, :, :, :]
        indices = np.array([y, y+patchsize, width-patchsize, width])
        patch_squence.append(patch)
        indices_squence.append(indices)
    
    for x in range(0, width-patchsize+1, stride):
        patch = inLF[:, height-patchsize:, x:x+patchsize, :, :, :]
        indices = np.array([height-patchsize, height, x, x+patchsize])
        patch_squence.append(patch)
        indices_squence.append(indices)
    
    patch = inLF[:, height-patchsize:, width-patchsize:, :, :, :]
    indices = np.array([height-patchsize, height, width-patchsize, width])
    patch_squence.append(patch)
    indices_squence.append(indices)
            
    return patch_squence, np.array(indices_squence)


def shaved_LF_reconstruct(patch_sequence, indices_sequence, border=(3, 3)):
    """
    Reconstruction function to recover the original light field from
    multiple cropped patches.

    :param patch_sequence:   a sequence of patches
    :param indices_sequence: indices of the input sequence of patches
    :param border:           shaved border

    :return:                 shaved light field
    """
    _, height, _, width = indices_sequence[-1]
    batchsize, _, _, s, t, channels = patch_sequence[0].shape
    
    img4d = np.zeros([1, height, width, s, t, channels], dtype=np.float32)
    for i in range(len(patch_sequence)):
        h_start, h_off, w_start, w_off = indices_sequence[i]
        img4d[0, h_start+border[0]:h_off-border[0],
              w_start+border[1]:w_off-border[1], ...] = shave_batch_LFs(patch_sequence[i], border)

    img4d = shave_batch_LFs(img4d, border)
    return img4d

