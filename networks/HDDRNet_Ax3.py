from utils.layers import *
from utils.convolve4d import *
from vgg19.vgg19 import VGG19
from tool.log_config import *


class HDDRNet(object):
    '''
    The HDDRNet framework

    '''

    def __init__(self, inputs, targets, is_training, args):
        super(HDDRNet, self).__init__()
        self.channels = args.channels
        self.gamma_S = args.gamma_S
        self.gamma_A = args.gamma_A
        self.use_perceptual_loss = args.perceptual_loss
        self.verbose = args.verbose
        self.net_variables = None
        self.vgg = VGG19(None, None, None)
        self.PreRecons, self.Recons = self.build_model(inputs, is_training, reuse=False, verbose=self.verbose)
        self.loss = self.compute_loss(targets, self.PreRecons, self.Recons, self.use_perceptual_loss)

    def build_model(self, x, is_training, reuse, verbose):
        with tf.variable_scope('lfresnet', reuse=reuse):
            if self.verbose:
                logging.info('+{0:-^72}+'.format(''))
            with tf.variable_scope('conv4d1'):
                x = conv4d(x, self.channels, 64, kernel_size_1=3, kernel_size_2=5, padding='SAME',
                           trainable=True, verbose=verbose)
                x = leakyrelu(x, verbose=verbose)
            source = x
            # ===============================  Angular Information Recovery ============================ #
            # ======= residual block 1 ====== #
            with tf.variable_scope('residual_block1_1'):
                with tf.variable_scope('angular_block1a'):
                    x = conv4d(x, 64, 64, kernel_size_1=3, kernel_size_2=5, padding='SAME',
                               trainable=True, verbose=verbose)
                    x = AGBN(x, is_training, verbose=verbose)
                    x = leakyrelu(x, verbose=verbose)
                with tf.variable_scope('angular_block1b'):
                    x = conv4d(x, 64, 64, kernel_size_1=3, kernel_size_2=5, padding='SAME',
                               trainable=True, verbose=verbose)
                    x = AGBN(x, is_training, verbose=verbose)
                residual_out1 = tf.add_n([x, source])
                x = residual_out1
            # ======= residual block 2 ====== #
            with tf.variable_scope('residual_block1_2'):
                with tf.variable_scope('angular_block1a'):
                    x = conv4d(x, 64, 64, kernel_size_1=3, kernel_size_2=5, padding='SAME',
                               trainable=True, verbose=verbose)
                    x = AGBN(x, is_training, verbose=verbose)
                    x = leakyrelu(x, verbose=verbose)
                with tf.variable_scope('angular_block1b'):
                    x = conv4d(x, 64, 64, kernel_size_1=3, kernel_size_2=5, padding='SAME',
                               trainable=True, verbose=verbose)
                    x = AGBN(x, is_training, verbose=verbose)
                residual_out2 = tf.add_n([x, residual_out1, source])
                x = residual_out2
            # ======= residual block 3 ====== #
            with tf.variable_scope('residual_block1_3'):
                with tf.variable_scope('angular_block1a'):
                    x = conv4d(x, 64, 64, kernel_size_1=3, kernel_size_2=5, padding='SAME',
                               trainable=True, verbose=verbose)
                    x = AGBN(x, is_training, verbose=verbose)
                    x = leakyrelu(x, verbose=verbose)
                with tf.variable_scope('angular_block1b'):
                    x = conv4d(x, 64, 64, kernel_size_1=3, kernel_size_2=5, padding='SAME',
                               trainable=True, verbose=verbose)
                    x = AGBN(x, is_training, verbose=verbose)
                residual_out3 = tf.add_n([x, residual_out2, residual_out1, source])
                x = residual_out3
            with tf.variable_scope('conv4d2'):
                x = conv4d(x, 64, 64, kernel_size_1=3, kernel_size_2=5, padding='SAME',
                           trainable=True, verbose=verbose)
                x = AGBN(x, is_training, verbose=verbose)
                x = tf.add_n([x, source])
            # ============================= Upsampling ============================= #
            with tf.variable_scope('conv4d3'):
                x = conv4d(x, 64, 3*9*9, kernel_size_1=3, kernel_size_2=5, padding='SAME',
                           trainable=True, verbose=verbose)
                x = angular_pixel_shuffle(x, rv_in=3, rv_out=9, verbose=verbose)
                x = leakyrelu(x, verbose=verbose)

            # ====================== Spatial Details Recovery ====================== #
            with tf.variable_scope('conv4d4'):
                x = conv4d(x, 27, 64, kernel_size_1=3, kernel_size_2=3, padding='SAME',
                           trainable=True, verbose=verbose)
                source2 = x

            # ======= residual block 4 ====== #
            with tf.variable_scope('residual_block2_1'):
                with tf.variable_scope('angular_block1a'):
                    x = conv4d(x, 64, 64, kernel_size_1=3, kernel_size_2=3, padding='SAME',
                               trainable=True, verbose=verbose)
                    x = AGBN(x, is_training, verbose=verbose)
                    x = leakyrelu(x, verbose=verbose)
                with tf.variable_scope('angular_block1b'):
                    x = conv4d(x, 64, 64, kernel_size_1=3, kernel_size_2=3, padding='SAME',
                               trainable=True, verbose=verbose)
                    x = AGBN(x, is_training, verbose=verbose)
                residual_out21 = tf.add_n([x, source2])
            with tf.variable_scope('conv4out'):
                x = conv4d(residual_out21, 64, self.channels, kernel_size_1=3, kernel_size_2=3, padding='SAME',
                           trainable=True, verbose=verbose)
                pre_recons = x

            if self.verbose:
                logging.info('+{0:-^72}+'.format(''))

        self.net_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='lfresnet')
        return x, pre_recons

    def views_flatten(self, x):
        batch, w, h, s, t, c = x.get_shape().as_list()
        x = tf.transpose(x, (0, 3, 4, 1, 2, 5))
        x_flatten = tf.reshape(x, (batch*s*t, w, h, c))
        return x_flatten

    def compute_loss(self, labels, pre_recons, recons, use_perceptual_loss):
        def inference_content_loss(x, outputs):
            _, x_phi = self.vgg.build_model(
                x, tf.constant(False), False)  # First
            _, outputs_phi = self.vgg.build_model(
                outputs, tf.constant(False), True)  # Second

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