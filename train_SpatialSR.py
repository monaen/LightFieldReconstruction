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

import argparse
import glob
import random
from utils.utils import downsampling, batchmeanpsnr, batchmeanssim
from utils.augmentation import *

# logging configuration
from tool.log_config import *
log_config()
tf.logging.set_verbosity(tf.logging.ERROR)

# ============================== Experimental settings ============================== #
parser = argparse.ArgumentParser(description="HDDRNet Tensorflow Implementation")
parser.add_argument("--datadir", type=str, default="./data", help="The training and testing data path")
parser.add_argument("--lr_start", type=float, default=1e-5, help="The start learning rate")
parser.add_argument("--lr_beta1", type=float, default=0.5,
                    help="The exponential decay rate for the 1st moment estimates")
parser.add_argument("--batchSize", type=int, default=1, help="The batchsize of the input data")
parser.add_argument("--imageSize", type=int, default=96, help="Spatial size of the input light fields")
parser.add_argument("--viewSize", type=int, default=5, help="Angular size of the input light fields")
parser.add_argument("--channels", type=int, default=1,
                    help="Channels=1 means only the luma channel; Channels=3 means RGB channels (not supported)")
parser.add_argument("--verbose", default=False, action="store_true", help="Whether print the network structure or not")
parser.add_argument("--num_epoch", type=int, default=50, help="The total number of training epoch")
parser.add_argument("--start_epoch", type=int, default=0, help="The start epoch counting number")
parser.add_argument("--gamma_S", type=int, default=4, choices=[1, 2, 3, 4], help="Spatial downscaling factor")
parser.add_argument("--gamma_A", type=int, default=1, choices=[0, 1, 2, 3, 4],
                    help="Angular downscaling factor, '0' represents 3x3->7x7")
parser.add_argument("--num_GRL_HRB", type=int, default=5, help="The number of HRB in GRLNet (only for AAAI model)")
parser.add_argument("--num_SRe_HRB", type=int, default=3, help="The number of HRB in SReNet (only for AAAI model)")
parser.add_argument("--resume", default=False, action="store_true", help="Need to resume the pretrained model or not")
parser.add_argument("--select_gpu", type=str, default="0", help="Select the gpu for training or evaluation")
parser.add_argument("--perceptual_loss", default=True, action="store_true",
                    help="Need to use perceptual loss or not, if true, one also have to set the vgg_model item")
parser.add_argument("--vgg_model", type=str, default="vgg19/weights/latest", help="Pretrained VGG model path")
parser.add_argument("--save_folder", type=str, default="checkpoints", help="model save path")
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


def adjust_learning_rate(learning_rate, epoch, step=20):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = learning_rate * (1 ** (epoch // step))
    return lr


def get_state(spatial_scale, angular_scale):
    statetype = ""
    if spatial_scale != 1:
        statetype += "Sx{:d}".format(spatial_scale)
    if angular_scale != 1:
        statetype += "Ax{:d}".format(angular_scale)
    return statetype


def save_model(sess, savefolder, epoch):
    if not os.path.exists(savefolder):
        os.makedirs(savefolder)
    savepath = os.path.join(savefolder, "epoch_{:03d}".format(epoch))
    saver = tf.train.Saver()
    path = saver.save(sess, savepath)
    return path


def save_stateinfo(save_folder, info_dict):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    statefile = open(os.path.join(save_folder, "state.txt"), "w")
    epoch = info_dict["epoch"]
    BESTPSNR = info_dict["BESTPSNR"]
    BESTSSIM = info_dict["BESTSSIM"]
    TestAvgPSNR = info_dict["TestAvgPSNR"]
    TestAvgSSIM = info_dict["TestAvgSSIM"]
    TestAvgAngularLoss = info_dict["TestAvgAngularLoss"]
    TestAvgSpatialLoss = info_dict["TestAvgSpatialLoss"]
    TestAvgTotalLoss = info_dict["TestAvgTotalLoss"]
    statefile.write("Epoch: {}\n".format(epoch))
    statefile.write("BESTPSNR: {}\n".format(BESTPSNR))
    statefile.write("BESTSSIM: {}\n".format(BESTSSIM))
    statefile.write("TestAvgPSNR: {}\n".format(TestAvgPSNR))
    statefile.write("TestAvgSSIM: {}\n".format(TestAvgSSIM))
    statefile.write("TestAvgAngularLoss: {}\n".format(TestAvgAngularLoss))
    statefile.write("TestAvgSpatialLoss: {}\n".format(TestAvgSpatialLoss))
    statefile.write("TestAvgTotalLoss: {}\n".format(TestAvgTotalLoss))
    statefile.close()
    

def read_stateinfo(save_folder):
    if os.path.exists(os.path.join(save_folder, "state.txt")):
        savedstate = open(os.path.join(save_folder, "state.txt"), "r")
        items = savedstate.read().splitlines()
        Epoch = np.int(items[0].split(":")[-1])
        BESTPSNR = np.float(items[1].split(":")[-1])
        BESTSSIM = np.float(items[2].split(":")[-1])
    else:
        logging.info("State Not Found. Initialize the training parameters")
        Epoch = 0
        BESTPSNR = 0.0
        BESTSSIM = 0.0
    return Epoch, BESTPSNR, BESTSSIM


def main(args):

    # ============ Setting the GPU used for model training ============ #
    logging.info("===> Setting the GPUs: {}".format(args.select_gpu))
    os.environ["CUDA_VISIBLE_DEVICES"] = args.select_gpu
    
    # ===================== Definition of params ====================== #
    logging.info("===> Initialization")
    inputs = tf.placeholder(tf.float32, [args.batchSize, args.imageSize//args.gamma_S, args.imageSize//args.gamma_S,
                                         args.viewSize, args.viewSize, args.channels])
    groundtruth = tf.placeholder(tf.float32, [args.batchSize, args.imageSize, args.imageSize, args.viewSize,
                                              args.viewSize, args.channels])
    is_training = tf.placeholder(tf.bool, [])
    learning_rate = tf.placeholder(tf.float32, [])
    
    HDDRNet = import_model(args.gamma_S, args.gamma_A)
    model = HDDRNet(inputs, groundtruth, is_training, args, state="TRAIN")

    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
    opt = tf.train.AdamOptimizer(beta1=args.lr_beta1, learning_rate=learning_rate)
    train_op = opt.minimize(model.loss, var_list=model.net_variables)

    init = tf.global_variables_initializer()
    sess.run(init)

    # ============ Restore the VGG-19 network ============ #
    if args.perceptual_loss:
        logging.info("===> Restoring the VGG-19 Network for Perceptual Loss")
        var = tf.global_variables()
        vgg_var = [var_ for var_ in var if "vgg19" in var_.name]
        saver = tf.train.Saver(vgg_var)
        saver.restore(sess, args.vgg_model)

    # ============ Load the Train / Test Data ============ #
    logging.info("===> Loading the Training and Test Datasets")
    trainlist = glob.glob(os.path.join(args.datadir, "MSTrain/5x5/*/*.npy"))
    testlist = glob.glob(os.path.join(args.datadir, "MSTest/5x5/*.npy"))

    BESTPSNR = 0.0
    BESTSSIM = 0.0
    statetype = get_state(args.gamma_S, args.gamma_A)
    
    # =========== Restore the pre-trained model ========== #
    if args.resume:
        logging.info("Resuming the pre-trained model.")
        Epoch, BESTPSNR, BESTSSIM = read_stateinfo(os.path.join(args.save_folder, statetype))
        saver = tf.train.Saver()
        try:
            saver.restore(sess, os.path.join(args.save_folder, statetype, "epoch_{:03d}".format(Epoch)))
            args.start_epoch = Epoch + 1
        except:
            logging.info("No saved model found.")
            args.start_epoch = 0

    logging.info("===> Start Training")
    
    for epoch in range(args.start_epoch, args.num_epoch):
        random.shuffle(trainlist)

        num_iter = len(trainlist) // args.batchSize
        lr = adjust_learning_rate(args.lr_start, epoch, step=20)

        for ii in range(num_iter):
            y_batch = np.load(trainlist[ii])
            x_batch = downsampling(y_batch, rs=args.gamma_S, ra=args.gamma_A, nSig=1.2)

            y_batch = y_batch.astype(np.float32) / 255.
            x_batch = x_batch.astype(np.float32) / 255.

            angular_loss = 0.0
            spatial_loss = 0.0
            total_loss = 0.0
            for j in range(len(y_batch)):
                x = np.expand_dims(x_batch[j], axis=0)
                y = np.expand_dims(y_batch[j], axis=0)

                _, aloss, sloss, tloss, recons = sess.run([train_op, model.angular_loss, model.spatial_loss, model.loss, model.Recons],
                                                          feed_dict={inputs: x, groundtruth: y, is_training: True, learning_rate: lr})

                angular_loss += aloss
                spatial_loss += sloss
                total_loss += tloss

            angular_loss /= len(y_batch)
            spatial_loss /= len(y_batch)
            total_loss /= len(y_batch)

            logging.info("Epoch {:03d} [{:03d}/{:03d}] |TRAIN|  Angular loss: {:.6f} | Spatial loss: {:.6f} | "
                         "Total loss: {:.6f} | Learning rate: {:.10f}".format(epoch, ii, num_iter, angular_loss,
                                                                              spatial_loss, total_loss, lr))

        # ===================== Testing ===================== #

        logging.info("===> Start Testing for Epoch {:03d}".format(epoch))
        num_testiter = len(testlist) // args.batchSize
        test_psnr = 0.0
        test_ssim = 0.0
        test_angularloss = []
        test_spatialloss = []
        test_totalloss = []

        for kk in range(num_testiter):
            y_batch = np.load(testlist[kk])
            x_batch = downsampling(y_batch, rs=args.gamma_S, ra=args.gamma_A, nSig=1.2)

            y_batch = y_batch.astype(np.float32) / 255.
            x_batch = x_batch.astype(np.float32) / 255.

            angular_loss = 0.0
            spatial_loss = 0.0
            total_loss = 0.0
            recons_batch = []
            for k in range(len(y_batch)):
                x = np.expand_dims(x_batch[k], axis=0)
                y = np.expand_dims(y_batch[k], axis=0)

                _, aloss, sloss, tloss, recons = sess.run([train_op, model.angular_loss, model.spatial_loss,
                                                           model.loss, model.Recons],
                                                          feed_dict={inputs: x,
                                                                     groundtruth: y,
                                                                     is_training: False,
                                                                     learning_rate: lr})

                angular_loss += aloss
                spatial_loss += sloss
                total_loss += tloss
                recons_batch.append(recons)

            angular_loss /= len(y_batch)  # average value for a single LF image
            spatial_loss /= len(y_batch)  # average value for a single LF image
            total_loss /= len(y_batch)  # average value for a single LF image

            logging.info("Epoch {:03d} [{:03d}/{:03d}] |TEST|  Angular loss: {:.6f} | Spatial loss: {:.6f} | "
                         "Total loss: {:.6f}".format(epoch, kk, num_testiter, angular_loss, spatial_loss, total_loss))

            recons_batch = np.concatenate(recons_batch, axis=0)
            recons_batch[recons_batch > 1.] = 1.
            recons_batch[recons_batch < 0.] = 0.
            item_psnr = batchmeanpsnr(y_batch, recons_batch)  # average value for a single LF image
            item_ssim = batchmeanssim(y_batch, recons_batch)  # average value for a single LF image

            test_angularloss.append(angular_loss)
            test_spatialloss.append(spatial_loss)
            test_totalloss.append(total_loss)
            test_psnr += item_psnr
            test_ssim += item_ssim

        test_psnr = test_psnr / len(testlist)
        test_ssim = test_ssim / len(testlist)
        avgtest_aloss = np.mean(test_angularloss)
        avgtest_sloss = np.mean(test_spatialloss)
        avgtest_tloss = np.mean(test_totalloss)
        test_dict = {"epoch": epoch,
                     "TestAvgPSNR": test_psnr,
                     "TestAvgSSIM": test_ssim,
                     "TestAvgAngularLoss": avgtest_aloss,
                     "TestAvgSpatialLoss": avgtest_sloss,
                     "TestAvgTotalLoss": avgtest_tloss,
                     "BESTPSNR": BESTPSNR,
                     "BESTSSIM": BESTSSIM}
        
        if test_psnr > BESTPSNR:
            savefolder = os.path.join(args.save_folder, statetype, "BESTPSNR")
            path = save_model(sess, savefolder, epoch)
            test_dict["BESTPSNR"] = test_psnr
            save_stateinfo(savefolder, test_dict)
            logging.info("Model saved to {}".format(path))
            logging.info("PSNR: {:.6f}(previous) update to {:.6f}(current) "
                         "[BEST PSNR weights saved]".format(BESTPSNR, test_psnr))
            BESTPSNR = test_psnr

        if test_ssim > BESTSSIM:
            savefolder = os.path.join(args.save_folder, statetype, "BESTSSIM")
            path = save_model(sess, savefolder, epoch)
            test_dict["BESTSSIM"] = test_ssim
            save_stateinfo(savefolder, test_dict)
            logging.info("Model saved to {}".format(path))
            logging.info("SSIM: {:.6f}(previous) update to {:.6f}(current) "
                         "[BEST SSIM weights saved]".format(BESTSSIM, test_ssim))
            BESTSSIM = test_ssim

        # =================== Save the epoch training info ===================== #
        path = save_model(sess, os.path.join(args.save_folder, statetype), epoch)
        save_stateinfo(os.path.join(args.save_folder, statetype), test_dict)
        logging.info("Model saved to {}".format(path))


if __name__ == "__main__":
    main(args)
