from utils.layers import *
from utils.convolve4d import *
from vgg19.vgg19 import VGG19
from tool.log_config import *


def ConvBNLReLU(x, in_channels, out_channels, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True,
                is_training=True, name="Conv4DBNLReLU"):
    with tf.variable_scope(name):
        x = conv4d(x, in_channels, out_channels, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                   padding=padding, trainable=trainable)
        x = AGBN(x, is_training)
        out = leakyrelu(x)
    return out


def ConvLReLU(x, in_channels, out_channels, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True,
              name="Conv4DLReLU"):
    with tf.variable_scope(name):
        x = conv4d(x, in_channels, out_channels, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                   padding=padding, trainable=trainable)
        out = leakyrelu(x)
    return out


def ConvBN(x, in_channels, out_channels, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True,
           is_training=True, name="Conv4DBN"):
    with tf.variable_scope(name):
        x = conv4d(x, in_channels, out_channels, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                   padding=padding, trainable=trainable)
        out = AGBN(x, is_training)
    return out


def HRB(x, in_channels, out_channels, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True,
        is_training=True, name="HRB"):
    with tf.variable_scope(name):
        out = ConvBNLReLU(x, in_channels, out_channels, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                          padding=padding, trainable=trainable, is_training=is_training)
        out = ConvBN(out, in_channels, out_channels, kernel_size_1=kernel_size_1, kernel_size_2=kernel_size_2,
                     padding=padding, trainable=trainable, is_training=is_training)
        out = tf.add(out, x)
    return out


def SpatialPixelShuffle(x, sr):
    batchsize, in_height, in_width, in_sview, in_tview, channels = x.get_shape().as_list()
    out_channels = channels / (sr ** 2)
    out_height = in_height * sr
    out_width = in_width * sr
    out_sview = in_sview
    out_tview = in_tview

    out = tf.transpose(x, [0, 5, 1, 2, 3, 4])
    out = tf.reshape(out, [batchsize, out_channels, sr, sr, in_height, in_width, in_sview, in_tview])
    out = tf.transpose(out, [0, 1, 4, 2, 5, 3, 6, 7])
    out = tf.reshape(out, [batchsize, out_channels, out_height, out_width, out_sview, out_tview])
    out = tf.transpose(out, [0, 2, 3, 4, 5, 1])
    return out


def AngularPixelShuffle(x, ar):
    batchsize, in_height, in_width, in_sview, in_tview, channels = x.get_shape().as_list()
    out_channels = channels / (ar ** 2)
    out_sview = in_sview * ar
    out_tview = in_tview * ar

    out = tf.transpose(x, [0, 1, 2, 5, 3, 4])
    out = tf.reshape(out, [batchsize, in_height, in_width, out_channels, ar, ar, in_sview, in_tview])
    out = tf.transpose(out, [0, 1, 2, 3, 6, 4, 7, 5])
    out = tf.reshape(out, [batchsize, in_height, in_width, out_channels, out_sview, out_tview])
    return out


def Upscaling_Spatial_and_Angular(x, ar=2, sr=2):
    sh = x.get_shape().as_list()
    dim = len(sh[1:-1])
    out = tf.reshape(x, [-1] + sh[-dim:])
    for i in range(dim, 2, -1):
        out = tf.concat([out, out], i)
    out_size = [-1] + [s for s in sh[1:3]] + [s * ar for s in sh[3:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size)

    if tf.shape(out)[4] % 2 == 0:
        out = out[:, :, :, 1:, 1:, :]

    out = SpatialPixelShuffle(out, sr)
    return out


def UpNet(x, sr=2, ar=2):
    if ar == 1:
        if sr == 2 or sr == 3:
            x = conv4d(x, 64, 64*sr*sr, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True)
            out = SpatialPixelShuffle(x, sr)
        if sr == 4:
            with tf.variable_scope("4x_01"):
                x = conv4d(x, 64, 64*2*2, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True)
                x = SpatialPixelShuffle(x, 2)
            with tf.variable_scope("4x_02"):
                x = conv4d(x, 64, 64*2*2, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True)
                out = SpatialPixelShuffle(x, 2)
    elif sr == 1:
        out = AngularPixelShuffle(x, ar)
    else:
        out = Upscaling_Spatial_and_Angular(x, ar, sr)
    return out


# ===================================================================================== #
#                             HDDRNet Framework (AAAI version)                          #
# ===================================================================================== #
class HDDRNet(object):
    '''
    The HDDRNet framework (AAAI version)
    '''

    def __init__(self, inputs, targets, is_training, args, state="TRAIN"):
        super(HDDRNet, self).__init__()
        self.channels = args.channels
        self.gamma_S = args.gamma_S
        self.gamma_A = args.gamma_A
        self.verbose = args.verbose
        self.net_variables = None
        self.num_GRL_HRB = args.num_GRL_HRB
        self.num_SRe_HRB = args.num_SRe_HRB
        self.GRLfeats = []
        self.SRefeats = []
        self.vgg = VGG19(None, None, None)
        self.PreRecons, self.Recons = self.build_model(inputs, is_training, reuse=False, verbose=self.verbose)
        if state == "TRAIN":
            self.use_perceptual_loss = args.perceptual_loss
            self.loss = self.compute_loss(targets, self.PreRecons, self.Recons, self.use_perceptual_loss)
        elif state == "TEST":
            self.use_perceptual_loss = None
            self.loss = 0.0
        else:
            assert False, "State not supportable (only support 'TRAIN' and 'TEST')."

    def build_model(self, x, is_training, reuse, verbose):
        with tf.variable_scope('lfresnet', reuse=reuse):
            if self.verbose:
                logging.info('+{0:-^72}+'.format(''))
            with tf.variable_scope('conv4d1'):
                x = conv4d(x, self.channels, 64, kernel_size_1=3, kernel_size_2=5, padding='SAME',
                           trainable=True, verbose=verbose)
                x = leakyrelu(x, verbose=verbose)
            SFEfeat1 = x
            # =============================  Geometric Representation Learning Net (GRLNet) ========================== #
            with tf.variable_scope("GRLNet"):
                for i in range(self.num_GRL_HRB):
                    out = HRB(x, 64, 64, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True,
                              is_training=is_training, name="RDB_{0:02d}".format(i))
                    x = x + out
                    self.GRLfeats.append(out)
            x = tf.add(SFEfeat1, x)

            with tf.variable_scope("UPNet"):
                x = UpNet(x, self.gamma_S, self.gamma_A)

            with tf.variable_scope("PreRecons"):
                x = ConvLReLU(x, 64, 64, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True,
                              name="Conv4DReLU")
                x = conv4d(x, 64, self.channels, kernel_size_1=1, kernel_size_2=1, padding="SAME", trainable=True)
            pre_recons = x

            # ==================================== Spatial Refine Network (SReNet) =================================== #
            with tf.variable_scope("SFENet_2"):
                x = conv4d(x, self.channels, 64, kernel_size_1=3, kernel_size_2=5, padding="SAME", trainable=True)
            SFEfeat2 = x

            with tf.variable_scope("SReNet"):
                for i in range(self.num_SRe_HRB):
                    out = HRB(x, 64, 64, kernel_size_1=3, kernel_size_2=3, padding="SAME", trainable=True,
                              is_training=is_training, name="RDB_{0:02d}".format(i))
                    x = x + out
                    self.SRefeats.append(out)
            x = tf.add(SFEfeat2, x)

            with tf.variable_scope("Output"):
                recons = conv4d(x, 64, self.channels, kernel_size_1=3, kernel_size_2=3, padding='SAME', trainable=True)

            if self.verbose:
                logging.info('+{0:-^72}+'.format(''))

        self.net_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lfresnet')
        return pre_recons, recons

    def views_flatten(self, x):
        batch, w, h, s, t, c = x.get_shape().as_list()
        x = tf.transpose(x, (0, 3, 4, 1, 2, 5))
        x_flatten = tf.reshape(x, (batch*s*t, w, h, c))
        return x_flatten

    def compute_loss(self, labels, pre_recons, recons, use_perceptual_loss):
        def inference_content_loss(x, outputs):
            _, x_phi = self.vgg.build_model(
                x, tf.constant(False), False) # First
            _, outputs_phi = self.vgg.build_model(
                outputs, tf.constant(False), True) # Second

            content_loss = None
            for i in range(len(x_phi)):
                l2_loss = tf.nn.l2_loss(x_phi[i] - outputs_phi[i])
                if content_loss is None:
                    content_loss = l2_loss
                else:
                    content_loss = content_loss + l2_loss
            return tf.reduce_mean(content_loss)

        def mse(labels, pre_recons):
            angular_loss = tf.reduce_mean(tf.keras.losses.MSE(pre_recons, labels), name='angular_loss')
            return angular_loss

        self.angular_loss = mse(labels, pre_recons)

        if use_perceptual_loss:
            self.content_loss = inference_content_loss(tf.tile(self.views_flatten(labels), (1, 1, 1, 3)),
                                                       tf.tile(self.views_flatten(recons), (1, 1, 1, 3)))
        else:
            self.content_loss = tf.constant(0.0)

        self.spatial_loss = 1e-5 * self.content_loss

        losses = self.spatial_loss + self.angular_loss

        return losses