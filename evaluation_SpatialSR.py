#!/usr/bin/env python
# ============================================================================================================= #
# |                                      HDDRNet (Tensorflow r1.*>=r1.8)                                      | #
# | Description:                                                                                              | #
# |     An Tensorflow implementation of "High-Dimensional Dense Residual Convolutional Neural Network         | #
# |     for Light Field Reconstruction".                                                                      | #
# |                                                                                                           | #
# | Paper:                                                                                                    | #
# |     High-Dimensional Dense Residual Convolutional Neural Network for Light Field Reconstruction           | #
# |     Nan Meng, Hayden Kwok-Hay So, Xing Sun, and Edmund Y. Lam                                             | #
# |     IEEE Transactions on Pattern Analysis and Machine Intelligence (T-PAMI), 2019.                        | #
# |                                                                                                           | #
# |     High-order Residual Network for Light Field Super-Resolution                                          | #
# |     Nan Meng, Xiaofei Wu, Jianzhuang Liu, Edmund Y. Lam                                                   | #
# |     The 34th AAAI Conference on Artificial Intelligence, 2020.                                            | #
# |                                                                                                           | #
# | Contact:                                                                                                  | #
# |     author: Nan Meng                                                                                      | #
# |     email:  u3003637@connect.hku.hk   |   nanmeng.uestc@hotmail.com                                       | #
# ============================================================================================================= #

import tensorflow as tf
import numpy as np
import argparse
import sys
import cv2
import scipy.io as sio
from tqdm import tqdm
from utils.utils import ssim_exact, downsampling, LF_split_patches, shaveLF, shave_batch_LFs, \
    shaveLF_by_factor, shaved_LF_reconstruct
from skimage.measure import compare_psnr as psnr

# logging configuration
from tool.log_config import *
log_config()
tf.logging.set_verbosity(tf.logging.ERROR)

# ============================== Experimental settings ============================== #
parser = argparse.ArgumentParser(description="HDDRNet Tensorflow Implementation")
parser.add_argument("--datapath", type=str, default="./data/evaluation/buddha.mat", help="The evaluation data path")
parser.add_argument("--batchSize", type=int, default=1, help="The batchsize of the input data")
parser.add_argument("--imageSize", type=int, default=96, help="Spatial size of the input light fields")
parser.add_argument("--viewSize", type=int, default=5, help="Angular size of the input light fields")
parser.add_argument("--channels", type=int, default=1,
                    help="Channels=1 means only the luma channel; Channels=3 means RGB channels (not supported)")
parser.add_argument("--verbose", default=True, action="store_true", help="Whether print the network structure or not")
parser.add_argument("--gamma_S", type=int, default=4, choices=[1, 2, 3, 4], help="Spatial downscaling factor")
parser.add_argument("--gamma_A", type=int, default=1, choices=[0, 1, 2, 3, 4],
                    help="Angular downscaling factor, '0' represents 3x3->7x7")
parser.add_argument("--num_GRL_HRB", type=int, default=5, help="The number of HRB in GRLNet (only for AAAI model)")
parser.add_argument("--num_SRe_HRB", type=int, default=3, help="The number of HRB in SReNet (only for AAAI model)")
parser.add_argument("--pretrained_model", type=str, default="pretrained_models/HDDRNet/Sx4/HDDRNet",
                    help="Path to store the pretrained model.")
parser.add_argument("--select_gpu", type=str, default="3", help="Select the gpu for training or evaluation")
args = parser.parse_args()


def import_model(scale_S, scale_A):
    if scale_A == 1:
        if scale_S == 4:
            from networks.HDDRNet_Sx4 import HDDRNet
        if scale_S == 3:
            from networks.HDDRNet_Sx3 import HDDRNet
        if scale_S == 2:
            from networks.HDDRNet_Sx2 import HDDRNet
    elif scale_S == 1:
        if scale_A == 0:
            from networks.HDDRNet_A3x3_7x7 import HDDRNet  # 3x3 -> 7x7
        if scale_A == 2:
            from networks.HDDRNet_Ax2 import HDDRNet  # 5x5 -> 9x9
        if scale_A == 3:
            from networks.HDDRNet_Ax3 import HDDRNet  # 3x3 -> 9x9
        if scale_A == 4:
            from networks.HDDRNet_Ax4 import HDDRNet  # 2x2 -> 8x8
    else:
        if scale_A == 2 and scale_S == 2:
            from networks.HDDRNet_Sx2Ax2 import HDDRNet
    return HDDRNet


def get_state(spatial_scale, angular_scale):
    statetype = ""
    if spatial_scale != 1:
        statetype += "Sx{:d}".format(spatial_scale)
    if angular_scale != 1:
        statetype += "Ax{:d}".format(angular_scale)
    return statetype


def ApertureWisePSNR(Groundtruth, Reconstruction):
    """
    Calculate the PSNR value for each sub-aperture image of the
    input reconstructed light field.

    :param Groundtruth:    input groundtruth light field
    :param Reconstruction: input reconstruced light field

    :return:               aperture-wise PSNR values
    """
    h, w, s, t = Groundtruth.shape[:4]
    PSNRs = np.zeros([s, t])
    for i in range(s):
        for j in range(t):
            gtimg = Groundtruth[:, :, i, j, ...]
            gtimg = np.squeeze(gtimg)
            recons = Reconstruction[:, :, i, j, ...]
            recons = np.squeeze(recons)
            PSNRs[i, j] = psnr(gtimg, recons)
    return PSNRs


def ApertureWiseSSIM(Groundtruth, Reconstruction):
    """
    Calculate the SSIM value for each sub-aperture image of the
    input reconstructed light field.

    :param Groundtruth:    input groundtruth light field
    :param Reconstruction: input reconstruced light field

    :return:               aperture-wise SSIM values
    """

    h, w, s, t = Groundtruth.shape[:4]
    SSIMs = np.zeros([s, t])
    for i in range(s):
        for j in range(t):
            gtimg = Groundtruth[:, :, i, j, ...]
            gtimg = np.squeeze(gtimg).astype(np.float32)/255.
            recons = Reconstruction[:, :, i, j, ...]
            recons = np.squeeze(recons).astype(np.float32)/255.
            SSIMs[i, j] = ssim_exact(gtimg, recons)
    return SSIMs


def get_indices(data, rs=4, patchsize=96, stride=30):
    """
    Calculate the indices for the splited LF patches.

    :param data:      input LF data
    :param rs:        upsampling factor
    :param patchsize: split size
    :param stride:    stride for spliting

    :return:          indices for all splited LF patches
    """
    b, h, w, s, t, c = data.shape
    recons_template = np.zeros([b, h*rs, w*rs, s, t, c])
    _, indices = LF_split_patches(recons_template, patchsize=patchsize, stride=stride)
    return indices


def ReconstructSpatialLFPatch(LFpatch, model, inputs, is_training, session, args, stride=60, border=(3, 3)):
    """
    Spatial reconstruct for a LF patch.

    :param LFpatch:     input LF patch
    :param model:       network
    :param inputs:      inputs of the network
    :param is_training: scala tensor to indicate whether need training
    :param session:     tensorflow session
    :param args:        arguments
    :param stride:      stride for reconstruction
    :param border:      shaved border of the final reconstructed LF patch

    :return:            reconstructed LF patch (border shaved)
    """
    recons_indices = get_indices(LFpatch, rs=args.gamma_S, patchsize=args.imageSize, stride=stride)
    inPatches, _ = LF_split_patches(LFpatch, patchsize=args.imageSize // args.gamma_S, stride=stride // args.gamma_S)
    reconPatches = []
    for i in tqdm(range(len(inPatches))):
        in_patch = inPatches[i]
        recon_patch = session.run(model.Recons, feed_dict={inputs: in_patch, is_training: False})
        reconPatches.append(recon_patch)

    reconsPatch_shaved = shaved_LF_reconstruct(reconPatches, recons_indices, border=border)
    return reconsPatch_shaved


def SpatialReconstruction(low_LF, model, inputs, is_training, session, args, stride=60, border=(3, 3)):

    # test stride values
    if stride % args.gamma_S != 0:
        stride = stride - np.mod(stride, args.gamma_S)

    # split the LF into 4D patches
    inLFpatch00 = low_LF[:, :, :, :5, :5, :]
    inLFpatch01 = low_LF[:, :, :, :5, -5:, :]
    inLFpatch10 = low_LF[:, :, :, -5:, :5, :]
    inLFpatch11 = low_LF[:, :, :, -5:, -5:, :]

    b, h, w, s, t, c = low_LF.shape
    ReconstructLF = np.zeros([b, h*args.gamma_S - border[0]*2, w*args.gamma_S - border[1]*2, s, t, c],
                             dtype=np.float32)

    # reconstruct each patch
    reconsLFPatch00 = ReconstructSpatialLFPatch(inLFpatch00, model, inputs, is_training,
                                                session, args, stride=stride, border=border)
    reconsLFPatch01 = ReconstructSpatialLFPatch(inLFpatch01, model, inputs, is_training,
                                                session, args, stride=stride, border=border)
    reconsLFPatch10 = ReconstructSpatialLFPatch(inLFpatch10, model, inputs, is_training,
                                                session, args, stride=stride, border=border)
    reconsLFPatch11 = ReconstructSpatialLFPatch(inLFpatch11, model, inputs, is_training,
                                                session, args, stride=stride, border=border)

    # stitch the LF patches together to get the final reconstruction LF
    ReconstructLF[:, :, :, :5, :5, :] = reconsLFPatch00
    ReconstructLF[:, :, :, :5, -5:, :] = reconsLFPatch01
    ReconstructLF[:, :, :, -5:, :5, :] = reconsLFPatch10
    ReconstructLF[:, :, :, -5:, -5:, :] = reconsLFPatch11

    ReconstructLF[ReconstructLF > 1.] = 1.
    ReconstructLF[ReconstructLF < 0.] = 0.
    return ReconstructLF


def LFUpsampling(inLF, scale=4, method="BICUBIC"):
    """
    Upsampling the input low-resolution light field by a given method.

    :param inLF:   input low-resolution light field
    :param scale:  upscaling factor
    :param method: given upscaling method

    :return:       upscaled light field
    """
    # check the inputs format
    assert isinstance(method, str), "The argument 'method' should be a string."

    # upsampling the input light field
    h, w, s, t = inLF.shape
    upscaledLF = np.zeros([h*scale, w*scale, s, t], dtype=np.uint8)
    for i in range(s):
        for j in range(t):
            lowimg = inLF[:, :, i, j]
            if method.lower() == "bicubic":
                upimg = cv2.resize(lowimg, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
            elif method.lower() == "bilinear":
                upimg = cv2.resize(lowimg, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
            elif method.lower() == "nearest":
                upimg = cv2.resize(lowimg, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
            else:
                assert False, "Upsampling method [{}] is not supportable.".format(method)
            upscaledLF[:, :, i, j] = upimg

    return upscaledLF


def main(args):
    # ============ Setting the GPU used for model training ============ #
    logging.info("===> Setting the GPUs: {}".format(args.select_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.select_gpu

    # ===================== Definition of params ====================== #
    logging.info("===> Initialization")
    inputs = tf.placeholder(tf.float32, [args.batchSize, args.imageSize // args.gamma_S, args.imageSize // args.gamma_S,
                                         args.viewSize, args.viewSize, args.channels])
    groundtruth = tf.placeholder(tf.float32, [args.batchSize, args.imageSize, args.imageSize, args.viewSize,
                                              args.viewSize, args.channels])
    is_training = tf.placeholder(tf.bool, [])

    HDDRNet = import_model(args.gamma_S, args.gamma_A)
    model = HDDRNet(inputs, groundtruth, is_training, args, state="TEST")

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    init = tf.global_variables_initializer()
    sess.run(init)

    # ================= Restore the pre-trained model ================= #
    logging.info("===> Resuming the pre-trained model.")
    saver = tf.train.Saver()
    try:
        saver.restore(sess, args.pretrained_model)
    except ValueError:
        logging.info("Pretrained model: {} not found.".format(args.pretrained_model))
        sys.exit(1)

    # ===================== Read light field data ===================== #
    logging.info("===> Reading the light field data")
    LF = sio.loadmat(args.datapath)["data"]
    LF = LF.transpose(2, 3, 0, 1, 4)
    LF = shaveLF_by_factor(LF, args.gamma_S)
    LF = np.expand_dims(LF, axis=0)
    Groundtruth = shave_batch_LFs(LF, border=(3, 3))
    Groundtruth = Groundtruth.squeeze()

    # ================== Downsample the light field =================== #
    logging.info("===> Downsampling")
    low_LF = downsampling(LF, rs=args.gamma_S, ra=args.gamma_A, nSig=1.2)
    low_inLF = low_LF.astype(np.float32) / 255.

    # ============= Reconstruct the original light field ============== #
    logging.info("===> Reconstructing ......")
    recons_LF = SpatialReconstruction(low_inLF, model, inputs, is_training, sess, args, stride=60, border=(3, 3))
    recons_LF = recons_LF.squeeze()
    recons_LF = np.uint8(recons_LF * 255.)

    logging.info("===> Calculating the mean PSNR and SSIM values (on luminance channel)......")
    meanPSNR = np.mean(ApertureWisePSNR(Groundtruth, recons_LF))
    meanSSIM = np.mean(ApertureWiseSSIM(Groundtruth, recons_LF))

    Bicubic = LFUpsampling(low_LF.squeeze(), scale=args.gamma_S, method="BICUBIC")
    Bicubic = shaveLF(Bicubic, border=(3, 3))
    meanbicubicPSNR = np.mean(ApertureWisePSNR(Groundtruth, Bicubic))
    meanbicubicSSIM = np.mean(ApertureWiseSSIM(Groundtruth, Bicubic))


    logging.info('{0:+^74}'.format(""))
    logging.info('|{0: ^72}|'.format("Quantitative result for the scene: {}".format(datapath.split('/')[-1])))
    logging.info('|{0: ^72}|'.format(""))
    logging.info('|{0: ^72}|'.format("Method: HDDRNet |  Mean PSNR: {:.3f}      Mean SSIM: {:.3f}".format(meanPSNR, meanSSIM)))
    logging.info('|{0: ^72}|'.format("Method: BICUBIC |  Mean PSNR: {:.3f}      Mean SSIM: {:.3f}".format(meanbicubicPSNR, meanbicubicSSIM)))
    logging.info('{0:+^74}'.format(""))


if __name__ == "__main__":
    main(args)
