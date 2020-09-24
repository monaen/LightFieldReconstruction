#!/usr/bin/env python
# ====================================================================================================================== #
# |                                      HDDRNet (Tensorflow r1.*>=r1.8)                                               | #
# | Description:                                                                                                       | #
# |     An Tensorflow implementation of "High-Dimensional Dense Residual Convolutional Neural Network                  | #
# |     for Light Field Reconstruction".                                                                               | #
# |                                                                                                                    | #
# | Citation:                                                                                                          | #
# |     @article{meng2019high,                                                                                         | #
# |              title={High-dimensional dense residual convolutional neural network for light field reconstruction},  | #
# |              author={Meng, Nan and So, Hayden Kwok-Hay and Sun, Xing and Lam, Edmund},                             | #
# |              journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},                             | #
# |              year={2019}                                                                                           | #
# |             }                                                                                                      | #
# |     @article{meng2020high,                                                                                         | #
# |              title={High-order residual network for light field super-resolution},                                 | #
# |              author={Meng, Nan and Wu, xiaofei and Liu, Jianzhuang and Lam, Edmund},                               | #
# |              journal={Association for the Advancement of Artificial Intelligence},                                 | #
# |              volume={34},                                                                                          | #
# |              number={7},                                                                                           | #
# |              pages={11757-11764},                                                                                  | #
# |              month={February},                                                                                     | #
# |              year={2020},                                                                                          | #
# |             }                                                                                                      | #
# | Contact:                                                                                                           | #
# |     author: Nan Meng                                                                                               | #
# |     email:  u3003637@connect.hku.hk   |   nanmeng.uestc@hotmail.com                                                | #
# ====================================================================================================================== #

import tensorflow as tf
import numpy as np
import argparse
import sys
import cv2
import scipy.io as sio
from tqdm import tqdm
from utils.utils import psnr, ssim_exact, downsampling, LF_split_patches, shaveLF, shave_batch_LFs, \
    shaveLF_by_factor, shaved_LF_reconstruct

# logging configuration
from tool.log_config import *
log_config()
tf.logging.set_verbosity(tf.logging.ERROR)

# ============================== Experimental settings ============================== #
parser = argparse.ArgumentParser(description="HDDRNet Tensorflow Implementation")
parser.add_argument("--datapath", type=str, default="./data/testset/occlusions20/occlusions_48.mat", help="The evaluation data path")
parser.add_argument("--batchSize", type=int, default=1, help="The batchsize of the input data")
parser.add_argument("--imageSize", type=int, default=96, help="Spatial size of the input light fields")
parser.add_argument("--viewSize", type=int, default=5, help="Angular size of the input light fields")
parser.add_argument("--channels", type=int, default=1,
                    help="Channels=1 means only the luma channel; Channels=3 means RGB channels (not supported)")
parser.add_argument("--verbose", default=True, action="store_true", help="Whether print the network structure or not")
parser.add_argument("--gamma_S", type=int, default=1, choices=[1, 2, 3, 4], help="Spatial downscaling factor")
parser.add_argument("--gamma_A", type=int, default=4, choices=[0, 1, 2, 3, 4],
                    help="Angular downscaling factor, '0' represents 3x3->7x7")
parser.add_argument("--num_GRL_HRB", type=int, default=5, help="The number of HRB in GRLNet (only for AAAI model)")
parser.add_argument("--num_SRe_HRB", type=int, default=3, help="The number of HRB in SReNet (only for AAAI model)")
parser.add_argument("--pretrained_model", type=str, default="pretrained_models/M-HDDRNet/Ax4/M-HDDRNet",
                    help="Path to store the pretrained model.")
parser.add_argument("--select_gpu", type=str, default="3", help="Select the gpu for training or evaluation")
args = parser.parse_args()


def import_model(scale_S, scale_A):
    """
    Network importation function.

    :param scale_S: spatial upsampling factor
    :param scale_A: angular upsampling factor

    :return:        network for the given super-resolution task.
    """
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
    """
    Get the super-resolution task.

    :param spatial_scale: spatial upsampling factor
    :param angular_scale: angular upsampling factor

    :return:              super-resolution task
    """
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


def main(args):
    # ============ Setting the GPU used for model training ============ #
    logging.info("===> Setting the GPUs: {}".format(args.select_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.select_gpu

    # ===================== Definition of params ====================== #
    logging.info("===> Initialization")
    if args.gamma_A == 0:    # 3x3 -> 7x7
        inputs = tf.placeholder(tf.float32, [None, None, None, 3, 3, args.channels])
        groundtruth = tf.placeholder(tf.float32, [None, None, None, 7, 7, args.channels])
    elif args.gamma_A == 2:  # 5x5 -> 9x9
        inputs = tf.placeholder(tf.float32, [None, None, None, 5, 5, args.channels])
        groundtruth = tf.placeholder(tf.float32, [None, None, None, 9, 9, args.channels])
    elif args.gamma_A == 3:  # 3x3 -> 9x9
        inputs = tf.placeholder(tf.float32, [None, None, None, 3, 3, args.channels])
        groundtruth = tf.placeholder(tf.float32, [None, None, None, 9, 9, args.channels])
    elif args.gamma_A == 4:  # 2x2 -> 8x8
        inputs = tf.placeholder(tf.float32, [None, None, None, 2, 2, args.channels])
        groundtruth = tf.placeholder(tf.float32, [None, None, None, 8, 8, args.channels])
    else:
        inputs = None
        groundtruth = None
    is_training = tf.placeholder(tf.bool, [])

    HDDRNet = import_model(args.gamma_S, args.gamma_A)
    model = HDDRNet(inputs, groundtruth, is_training, args, state="TEST")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)

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
    LF = LF.transpose(2, 3, 0, 1)
    LF = shaveLF(LF, border=(60, 60))
    LF = np.expand_dims(LF, axis=0)
    LF = np.expand_dims(LF, axis=-1)

    # ================== Downsample the light field =================== #
    logging.info("===> Downsampling")
    Groundtruth, low_LF = downsampling(LF, rs=args.gamma_S, ra=args.gamma_A, nSig=1.2)
    Groundtruth = Groundtruth.squeeze()
    low_inLF = low_LF.astype(np.float32) / 255.

    # ============= Reconstruct the original light field ============== #
    logging.info("===> Reconstructing ......")
    start_time = time.time()
    recons_LF = sess.run(model.Recons, feed_dict={inputs: low_inLF, is_training: False})
    logging.info("Execute Time: {0:.6f}".format(time.time() - start_time))

    recons_LF = recons_LF.squeeze()
    recons_LF = np.uint8(recons_LF * 255.)

    logging.info("===> Calculating the mean PSNR and SSIM values (on luminance channel)......")
    meanPSNR = np.mean(ApertureWisePSNR(Groundtruth, recons_LF))
    meanSSIM = np.mean(ApertureWiseSSIM(Groundtruth, recons_LF))

    logging.info("{0:+^74}".format(""))
    logging.info("|{0: ^72}|".format("Quantitative result for the scene: {}".format(args.datapath.split('/')[-1])))
    logging.info("|{0: ^72}|".format(""))
    logging.info("|{0: ^72}|".format("Method: HDDRNet |  Mean PSNR: {:.3f}      Mean SSIM: {:.3f}".format(meanPSNR, meanSSIM)))
    logging.info("{0:+^74}".format(""))


if __name__ == "__main__":
    main(args)
