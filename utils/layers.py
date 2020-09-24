import numpy as np
import sys
sys.path.append('utils')
from convolve4d import *
from tool.log_config import *


def initializer_fullyconnect(in_channel, out_channel, stddev_factor=1.0, mode='Glorot'):
    """Initialization in the style of Glorot 2010.
    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
    if mode == 'Glorot':
        stddev = np.sqrt(stddev_factor / np.sqrt(in_channel*out_channel))
    else:
        stddev = 1.0 # standard initialization
    return tf.truncated_normal([in_channel, out_channel], mean=0.0, stddev=stddev)


def initializer_conv2d(in_channels, out_channels, mapsize,
                       stddev_factor=1.0, mode='Glorot'):
    """Initialization in the style of Glorot 2010.
    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
    
    if mode == 'Glorot':
        stddev = np.sqrt(stddev_factor / (np.sqrt(in_channels*out_channels)*mapsize*mapsize))
    else:
        stddev = 1.0
    
    return tf.truncated_normal([mapsize, mapsize, in_channels, out_channels], mean=0.0, stddev=stddev)


def initializer_conv4d(in_channels, out_channels, mapsize, mapsize2,
                       stddev_factor=1.0, mode='Glorot'):
    """Initialization in the style of Glorot 2010.
    stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""
    if mode == 'Glorot':
        stddev = np.sqrt(stddev_factor / (np.sqrt(in_channels*out_channels)*mapsize*mapsize*mapsize2*mapsize2))
    else:
        stddev = 1.0

    return tf.truncated_normal([mapsize, mapsize, mapsize2, mapsize2, in_channels, out_channels],
                               mean=0.0, stddev=stddev)


# ======================================================================================= #
#                            Basic Modules for HDDRNet Framework                          #
# ======================================================================================= #

def conv4d(x, in_channels, out_channels, kernel_size_1=3, kernel_size_2=3, 
           stride_1=1, stride_2=1, padding='SAME', stddev_factor=1.0, trainable=True, verbose=True):

    assert len(x.get_shape().as_list()) == 6 and \
           "Previous layer must be 6D: (batch, height, width, sview, tview, channels)"
    
    weight4d_init = initializer_conv4d(in_channels, out_channels, mapsize=kernel_size_1, 
                                       mapsize2=kernel_size_2, stddev_factor=stddev_factor)
    
    filter4d = tf.get_variable(name='weight', 
                               initializer=weight4d_init, 
                               dtype=tf.float32, 
                               trainable=trainable)
    
    out = convolve4d(input=x, filter=filter4d, 
                     strides=[1, stride_1, stride_1, stride_2, stride_2, 1], 
                     padding=padding)

    if verbose:
        message = '|{0:-^72}|'.format(' Conv4d Layer: ' + str(out.get_shape()) + ' ')
        logging.info(message)
    return out


def relu(x, verbose=True):
    ''' ReLU: activation function
    '''
    out = tf.nn.relu(x)

    if verbose:
        message = '|{0:-^72}|'.format(' ReLU Layer: ' + str(out.get_shape()) + ' ')
        logging.info(message)
    return out


def elu(x, verbose=True):
    ''' ELU: activation function
    '''
    out = tf.nn.elu(x)

    if verbose:
        message = '|{0:-^72}|'.format(' ELU Layer: ' + str(out.get_shape()) + ' ')
        logging.info(message)
    return out


def leakyrelu(x, leak=0.2, verbose=True):
    '''Adds a leaky ReLU (LReLU) activation function to this model'''
    t1  = .5 * (1 + leak)
    t2  = .5 * (1 - leak)
    out = t1 * x + t2 * tf.abs(x)

    if verbose:
        message = '|{0:-^72}|'.format(' LeakyReLU Layer: ' + str(out.get_shape()) + ' ')
        logging.info(message)
    return out


def prelu(x, trainable=True, verbose=True):
    alpha = tf.get_variable(
        name='alpha', 
        shape=x.get_shape()[-1],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    out = tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

    if verbose:
        message = '|{0:-^72}|'.format(' PReLU Layer: ' + str(out.get_shape()) + ' ')
        logging.info(message)
    return out


def sigmoid(x, verbose=True):
    '''Append a sigmoid (0,1) activation function layer to the model.'''
    out = tf.nn.sigmoid(x)

    if verbose:
        message = '|{0:-^72}|'.format(' Sigmoid Layer: ' + str(out.get_shape()) + ' ')
        logging.info(message)
    return out


def AGBN(x, is_training, decay=0.99, epsilon=0.001, trainable=True, verbose=True):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2, 3, 4])
        train_mean = tf.assign(
            pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(
            pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(
                x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(
            x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=trainable)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=trainable)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var', 
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)

    if verbose:
        message = '|{0:-^72}|'.format(' AGBN Layer: ' + str(x.get_shape()) + ' ')
        logging.info(message)
    return tf.cond(is_training, bn_train, bn_inference)


def spacial_pixel_shuffle(x, rs, verbose=True):
    batchsize, in_height, in_width, in_sview, in_tview, channels = x.get_shape().as_list()
    out_channels = int(channels / (rs**2))
    out_height = int(in_height * rs)
    out_width = int(in_width * rs)
    out_sview = int(in_sview)
    out_tview = int(in_tview)

    out = tf.transpose(x, (0, 5, 1, 2, 3, 4))
    out = tf.reshape(out, (batchsize, out_channels, rs, rs, in_height, in_width, in_sview, in_tview))
    out = tf.transpose(out, (0, 1, 4, 2, 5, 3, 6, 7))
    out = tf.reshape(out, (batchsize, out_channels, out_height, out_width, out_sview, out_tview))
    out = tf.transpose(out, (0, 2, 3, 4, 5, 1))

    if verbose:
        message = '|{0:-^72}|'.format(' Spatial Pixel Shuffle Layer: ' + str(out.get_shape()) + ' ')
        logging.info(message)
    return out


def spatio_angular_upsampling(x, K=2, verbose=True):
    ''' Upscaling Layer: upscale the inputs by K times of the original size.
        For view use nearest neighboor
        For spacial use pixel shuffle
    '''

    # ========= angular pixel interpolation ========= #
    sh = x.get_shape().as_list()
    dim = len(sh[1:-1])
    out = (tf.reshape(x, [-1] + sh[-dim:]))
    for i in range(dim, 2, -1):
        out = tf.concat([out, out], i)
    out_size = [-1] + [s for s in sh[1:3]] + [s * 2 for s in sh[3:-1]] + [sh[-1]]
    out = tf.reshape(out, out_size, name='upsampling')

    if out.get_shape().as_list()[4] % 2 == 0:
        out = out[:, :, :, 1:, 1:, :]

    # ============ spacial pixel shuffle ============ #
    out = spacial_pixel_shuffle(out, K, verbose=False)

    if verbose:
        message = '|{0:-^72}|'.format(' Spatio-Angular Upsampling Layer: ' + str(out.get_shape()) + ' ')
        logging.info(message)
    return out


def angular_pixel_shuffle(x, rv_in, rv_out, verbose=True):
    batchsize, in_height, in_width, in_sview, in_tview, channels = x.get_shape().as_list()
    out_channels = int(channels / (rv_out**2))

    out = tf.transpose(x, (0, 1, 2, 5, 3, 4))
    shape = tf.shape(out)
    out = tf.reshape(out, [shape[0], shape[1], shape[2], out_channels, rv_out, rv_out, in_sview, in_tview])
    out = tf.transpose(out, (0, 1, 2, 3, 6, 4, 7, 5))
    out = tf.reshape(out, (shape[0], shape[1], shape[2], out_channels, rv_out*in_sview, rv_out*in_tview))
    out = tf.reshape(out, (shape[0], shape[1], shape[2], out_channels, int(rv_out*in_sview/rv_in), rv_in ,
                           int(rv_out*in_tview/rv_in), rv_in))
    out = tf.transpose(out, (0, 1, 2, 3, 5, 7, 4, 6))
    out = tf.reshape(out, (shape[0], shape[1], shape[2], out_channels*rv_in*rv_in, int(rv_out*in_sview/rv_in),
                           int(rv_out*in_tview/rv_in)))
    out = tf.transpose(out, (0, 1, 2, 4, 5, 3))

    if verbose:
        message = '|{0:-^72}|'.format(' Angular Upsampling Layer: ' + str(out.get_shape()) + ' ')
        logging.info(message)
    return out

# ======================================================================================= #
#                            Basic Modules for VGG Network                                #
# ======================================================================================= #
def conv_layer(x, filter_shape, stride, trainable=True):
    filter_ = tf.get_variable(
        name='weight',
        shape=filter_shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        trainable=trainable)
    return tf.nn.conv2d(
        input=x,
        filter=filter_,
        strides=[1, stride, stride, 1],
        padding='SAME')


def batch_normalize2d(x, is_training, decay=0.99, epsilon=0.001, trainable=True):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(
            pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(
            pop_var, pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(
                x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(
            x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=trainable)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=trainable)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)
    return tf.cond(is_training, bn_train, bn_inference)


def lrelu(x):
    alpha = 0.2
    return tf.maximum(alpha * x, x)


def max_pooling_layer(x, size, stride):
    return tf.nn.max_pool(
        value=x,
        ksize=[1, size, size, 1],
        strides=[1, stride, stride, 1],
        padding='SAME')


def flatten_layer(x):
    ''' Flatten: flat the outputs of the last layer.s
    '''
    input_shape = x.get_shape().as_list()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))

    return tf.reshape(transposed, [-1, dim])


def full_connection_layer(x, out_dim, trainable=True):
    in_dim = x.get_shape().as_list()[-1]
    W = tf.get_variable(
        name='weight',
        shape=[in_dim, out_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=trainable)
    b = tf.get_variable(
        name='bias',
        shape=[out_dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=trainable)
    return tf.add(tf.matmul(x, W), b)