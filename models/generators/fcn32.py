
import sys
import layers
import tensorflow as tf
import numpy as np
weights_path = "/gan_segmentation/vgg16_weights.npz"

def fcn32(x, pkeep, classes, pretrained = 1):

    weights = np.load(weights_path)

    if pretrained == 0:
        pretrained_vars = [0]*15
    else:
        pretrained_vars = [ \
            [weights["conv1_1_W"], weights["conv1_1_b"]],
            [weights["conv1_2_W"], weights["conv1_2_b"]],
            [weights["conv2_1_W"], weights["conv2_1_b"]],
            [weights["conv2_2_W"], weights["conv2_2_b"]],
            [weights["conv3_1_W"], weights["conv3_1_b"]],
            [weights["conv3_2_W"], weights["conv3_2_b"]],
            [weights["conv3_3_W"], weights["conv3_3_b"]],
            [weights["conv4_1_W"], weights["conv4_1_b"]],
            [weights["conv4_2_W"], weights["conv4_2_b"]],
            [weights["conv4_3_W"], weights["conv4_3_b"]],
            [weights["conv5_1_W"], weights["conv5_1_b"]],
            [weights["conv5_2_W"], weights["conv5_2_b"]],
            [weights["conv5_3_W"], weights["conv5_3_b"]],
            [weights["fc6_W"], weights["fc6_b"]],
            [weights["fc7_W"], weights["fc7_b"]]]

    with tf.variable_scope('generator'):

        # NHWC TO NCHW *************************************************************************************************

        x = tf.transpose(x, [0, 3, 1, 2])

        # DEFINE MODEL *************************************************************************************************

        print('Building generator model fcn32')
        print('\tBuilding unit: conv1')
        # Convolution 1
        x = layers.conv_layer(x, [3, 3, 3, 64], pkeep, "conv1_1", pretrained_vars[0], data_format='NCHW')
        # Convolution 2
        x = layers.conv_layer(x, [3, 3, 64, 64], pkeep, "conv1_2", pretrained_vars[1], data_format='NCHW')
        # Max-Pooling 1
        x = layers.max_pool_layer(x, padding='SAME', data_format='NCHW')
        print('\tBuilding unit: conv2')
        # Convolution 3
        x = layers.conv_layer(x, [3, 3, 64, 128], pkeep, "conv2_1", pretrained_vars[2], data_format='NCHW')
        # Convolution 4
        x = layers.conv_layer(x, [3, 3, 128, 128], pkeep, "conv2_2", pretrained_vars[3], data_format='NCHW')
        # Max-Pooling 2
        x = layers.max_pool_layer(x, padding='SAME', data_format='NCHW')
        print('\tBuilding unit: conv3')
        # Convolution 5
        x = layers.conv_layer(x, [3, 3, 128, 256], pkeep, "conv3_1", pretrained_vars[4], data_format='NCHW')
        # Convolution 6
        x = layers.conv_layer(x, [3, 3, 256, 256], pkeep, "conv3_2", pretrained_vars[5], data_format='NCHW')
        # Convolution 7
        x = layers.conv_layer(x, [3, 3, 256, 256], pkeep, "conv3_3", pretrained_vars[6], data_format='NCHW')
        # Max-Pooling 3
        x = layers.max_pool_layer(x, padding='SAME', data_format='NCHW')
        print('\tBuilding unit: conv4')
        # Convolution 8
        x = layers.conv_layer(x, [3, 3, 256, 512], pkeep, "conv4_1", pretrained_vars[7], data_format='NCHW')
        # Convolution 9
        x = layers.conv_layer(x, [3, 3, 512, 512], pkeep, "conv4_2", pretrained_vars[8], data_format='NCHW')
        # Convolution 10
        x = layers.conv_layer(x, [3, 3, 512, 512], pkeep, "conv4_3", pretrained_vars[9], data_format='NCHW')
        # Max-Pooling 4
        x = layers.max_pool_layer(x, padding='SAME', data_format='NCHW')
        print('\tBuilding unit: conv5')
        # Convolution 11
        x = layers.conv_layer(x, [3, 3, 512, 512], pkeep, "conv5_1", pretrained_vars[10], data_format='NCHW')
        # Convolution 12
        x = layers.conv_layer(x, [3, 3, 512, 512], pkeep, "conv5_2", pretrained_vars[11], data_format='NCHW')
        # Convolution 13
        x = layers.conv_layer(x, [3, 3, 512, 512], pkeep, "conv5_3", pretrained_vars[12], data_format='NCHW')
        # Max-Pooling 5
        x = layers.max_pool_layer(x, padding='SAME', data_format='NCHW')
        print('\tBuilding unit: fully conv')
        # Dense-Conv 1
        x = layers.conv_layer(x, [7, 7, 512, 4096], pkeep, "fc6", pretrained_vars[13], data_format='NCHW')
        # Dense-Conv 2
        x = layers.conv_layer(x, [1, 1, 4096, 4096], pkeep, "fc7", pretrained_vars[14], data_format='NCHW')
        # Score Predictions
        x = layers.conv_layer(x, [1, 1, 4096, classes], name="final", data_format='NCHW', relu='no')
        unscaled = tf.transpose(x, [0, 2, 3, 1])

        # Upsample
        size = tf.shape(x)
        height = size[1]
        width = size[2]
        logits = tf.image.resize_images(
            unscaled,
            tf.stack([height * 32, width * 32]),
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

        return unscaled, logits
