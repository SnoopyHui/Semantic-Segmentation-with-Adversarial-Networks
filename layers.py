
# Imports
from __future__ import print_function, division
import tensorflow as tf
from utils import identity_initializer
from utils import spectral_normed_weight
import numpy as np
import pdb

WEIGHT_DECAY_KEY = 'WEIGHT_DECAY' # l2 norm collection
SPECTRAL_NORM_UPDATE_OPS = "spectral_norm_update_ops"


def conv_layer(previous, filter_shape, pkeep = 1, name = "conv_layer", weights = 0, strides = 1, relu = 'yes', data_format='NHWC', padding='SAME'):
    if data_format == 'NCHW':
        strides = [1, 1, strides, strides]
    else:
        strides = [1, strides, strides, 1]
    with tf.variable_scope(name):
        if weights is not 0:
            kernel_values = weights[0]
            bias_values = weights[1]
            if tuple(filter_shape) != weights[0].shape:
                kernel_values = np.reshape(kernel_values, tuple(filter_shape))
            init = tf.constant_initializer(kernel_values, dtype=tf.float32)
            W_conv = tf.get_variable(initializer=init, shape=kernel_values.shape, name='w_conv')
        else:
            stddev = np.sqrt(2 / (filter_shape[0] ** 2 * filter_shape[3]))
            W_conv = tf.get_variable('kernel', filter_shape,
            tf.float32, initializer=tf.random_normal_initializer(stddev=stddev))
            bias_values = np.zeros(filter_shape[3])
        init = tf.constant_initializer(bias_values, dtype=tf.float32)
        B_conv = tf.get_variable(initializer=init,  shape=bias_values.shape, name='b_conv')
        if W_conv not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, W_conv)
        if B_conv not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, B_conv)
        if W_conv not in tf.get_collection(SPECTRAL_NORM_UPDATE_OPS):
            tf.add_to_collection(SPECTRAL_NORM_UPDATE_OPS, W_conv)
        if B_conv not in tf.get_collection(SPECTRAL_NORM_UPDATE_OPS):
            tf.add_to_collection(SPECTRAL_NORM_UPDATE_OPS, B_conv)
        conv = tf.nn.conv2d(previous, W_conv, strides=strides, padding=padding, data_format=data_format)
        conv = tf.nn.bias_add(conv, B_conv, data_format = data_format)
        if relu == 'yes':
            conv = tf.nn.relu(conv)
        convd = tf.nn.dropout(conv, pkeep)
    return convd

def deconv_layer(previous, filter_shape, name='deconv_layer', data_format='NHWC', padding='SAME'):
    with tf.variable_scope(name):
        previous_shape = tf.shape(previous)
        if data_format == 'NCHW':
            strides = [1, 1, 2, 2]
            output_shape = tf.stack([previous_shape[0], previous_shape[3] // 2, previous_shape[1] * 2,
                                     previous_shape[2] * 2])
        else:
            strides = [1, 2, 2, 1]
            output_shape = tf.stack([previous_shape[0], previous_shape[1] * 2, previous_shape[2] * 2,
                                     previous_shape[3] // 2])
        stddev = np.sqrt(2 / (filter_shape[0] ** 2 * filter_shape[3]))
        W_deconv = tf.get_variable(initializer = tf.truncated_normal_initializer(mean=0.0, stddev=stddev),
                                   shape = filter_shape, name ='W_deconv' )
        B_deconv = tf.get_variable(initializer=tf.zeros_initializer, shape = [filter_shape[-2]], name='B_deconv')
        if W_deconv not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, W_deconv)
        if B_deconv not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, B_deconv)
        deconv = tf.nn.conv2d_transpose(previous, filter=W_deconv, output_shape=output_shape, strides=strides,
                                        padding=padding, data_format=data_format)
        deconv = tf.nn.bias_add(deconv, B_deconv, data_format=data_format)
        deconv = tf.nn.relu(deconv)
    return deconv

def atrous_conv_layer(previous, filter_shape, output_stride, weights = 0, pkeep=1, name='atrous_conv', padding='SAME', identity_init = 'no'):
    with tf.variable_scope(name):
        if weights is not 0:
            kernel_values = weights[0]
            bias_values = weights[1]
            if tuple(filter_shape) != weights[0].shape:
                kernel_values = np.reshape(kernel_values, tuple(filter_shape))
            W_init = tf.constant_initializer(kernel_values, dtype=tf.float32)
            B_init = tf.constant_initializer(bias_values, dtype=tf.float32)
            W_atrous = tf.get_variable(initializer=W_init, shape=filter_shape, name='w_atrous')
            B_atrous = tf.get_variable(initializer=B_init, shape=filter_shape[-1], name='b_atrous')
        else:
            if identity_init == 'yes':
                kernel_values = identity_initializer(filter_shape)
                bias_values = np.zeros((filter_shape[-1]))
                W_init = tf.constant_initializer(kernel_values, dtype=tf.float32)
                B_init = tf.constant_initializer(bias_values, dtype=tf.float32)
                W_atrous = tf.get_variable(initializer=W_init, shape=filter_shape, name='w_atrous')
                B_atrous = tf.get_variable(initializer=B_init, shape=filter_shape[-1], name='b_atrous')
            else:
                stddev = np.sqrt(2 / (filter_shape[0] ** 2 * filter_shape[3]))
                W_atrous = tf.get_variable(initializer=tf.random_normal_initializer(stddev=stddev),
                                           shape = filter_shape, name = 'w_atrous')
                B_atrous = tf.get_variable(initializer=tf.zeros_initializer(), shape=[filter_shape[-1]], name = 'b_atrous')

        if W_atrous not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, W_atrous)
        if B_atrous not in tf.get_collection(WEIGHT_DECAY_KEY):
            tf.add_to_collection(WEIGHT_DECAY_KEY, B_atrous)

        atrous_conv = tf.nn.atrous_conv2d(previous, W_atrous, rate=output_stride, padding=padding)
        atrous_conv = tf.nn.bias_add(atrous_conv, B_atrous)
        atrous_conv = tf.nn.dropout(atrous_conv, pkeep)
        atrous_conv = tf.nn.relu(atrous_conv)
    return atrous_conv

def upsampling_layer(previous, name='upsampling', factor=32):
    with tf.variable_scope(name):
        # Deconvolution and Logits
        previous_shape = tf.shape(previous)
        logits = tf.image.resize_images(previous,
                                        tf.stack([previous_shape[2]*factor, previous_shape[2]*factor]),
                                        method=tf.image.ResizeMethod.BILINEAR,
                                        align_corners=False)
    return logits

def max_pool_layer(previous, data_format='NHWC', padding='SAME'):
    if data_format == 'NCHW':
        strides = [1, 1, 2, 2]
        ksize = [1, 1, 2, 2]
    else:
        strides = [1, 2, 2, 1]
        ksize = [1, 2, 2, 1]
    return tf.nn.max_pool(previous, ksize=ksize, strides=strides, padding=padding, data_format=data_format)

def avg_pool_layer(previous, data_format='NHWC', padding='SAME'):
    if data_format == 'NCHW':
        strides = [1, 1, 2, 2]
        ksize = [1, 1, 2, 2]
    else:
        strides = [1, 2, 2, 1]
        ksize = [1, 2, 2, 1]
    return tf.nn.avg_pool(previous, ksize=ksize, strides=strides, padding=padding, data_format=data_format)

def crop_and_concat_layer(x1,x2):
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    return tf.concat([x1_crop, x2], 3)