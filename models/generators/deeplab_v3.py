import tensorflow as tf
from layers import atrous_conv_layer, conv_layer
from utils import mybn, myrelu, mygn
import pdb

# RESNET MODULES *******************************************************************************************************

def first_residual_block(x, kernel, out_channel, strides, is_train, name="unit"):
    input_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        if input_channels == out_channel:
            if strides == 1:
                shortcut = tf.identity(x) # returns a tensor with the same shape and contents as x
            else:
                shortcut = tf.nn.max_pool(x, [1, strides, strides, 1], [1, strides, strides, 1], 'VALID')
        else:
            in_shape = x.get_shape()
            shortcut = conv_layer(x, [1, 1, in_shape[3], out_channel], strides=strides, name = "shortcut") # 1x1 conv to obtain out_channel maps
        # Residual
        in_shape = x.get_shape()
        x = conv_layer(x, [kernel, kernel, in_shape[3], out_channel], strides=strides, name = "conv_1")
        x = mygn(x, name='gn_1')
        x = myrelu(x, name='relu_1')
        in_shape = x.get_shape()
        x = conv_layer(x, [kernel, kernel, in_shape[3], out_channel], strides=1, name="conv_2")
        x = mygn(x, name='gn_2')
        # Merge
        x = x + shortcut
        x = myrelu(x, name='relu_2')
    return x

def residual_block(x, kernel, is_train, name="unit"):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = x
        # Residual
        in_shape = x.get_shape()
        x = conv_layer(x, [kernel, kernel, in_shape[3], num_channel], strides=1, name="conv_1")
        x = mygn(x, name='gn_1')
        x = myrelu(x, name='relu_1')
        in_shape = x.get_shape()
        conv_layer(x, [kernel, kernel, in_shape[3], num_channel], strides=1, name="conv_2")
        x = mygn(x, name='gn_2')
        # Merge
        x = x + shortcut
        x = myrelu(x, name='relu_2')
    return x

# DEEPLAB MODULES ******************************************************************************************************

def first_residual_atrous_block(x, kernel, out_channel, strides, is_train, name="unit"):
    input_channels = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        if input_channels == out_channel:
            if strides == 1:
                shortcut = tf.identity(x) # returns a tensor with the same shape and contents as x
            else:
                shortcut = tf.nn.max_pool(x, [1, 1, 1, 1], [1, 1, 1, 1], 'VALID')
        else:
            in_shape = x.get_shape()
            shortcut = atrous_conv_layer(x, [1, 1, in_shape[-1], out_channel], out_channel, strides, name='shortcut')  # 1x1 conv to obtain out_channel maps
        # Residual
        in_shape = x.get_shape()
        x = atrous_conv_layer(x, [kernel, kernel, in_shape[-1], out_channel], out_channel, strides, name='conv_1')
        x = mygn(x, name='gn_1')
        x = myrelu(x, name='relu_1')
        in_shape = x.get_shape()
        x = atrous_conv_layer(x, [kernel, kernel, in_shape[-1], out_channel], out_channel, strides, name='conv_2')
        x = mygn(x, name='gn_2')
        # Merge
        x = x + shortcut
        x = myrelu(x, name='relu_2')
    return x

def residual_atrous_block(x, kernel, is_train, name="unit"):
    num_channel = x.get_shape().as_list()[-1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding residual unit: %s' % scope.name)
        # Shortcut connection
        shortcut = x
        # Residual
        in_shape = x.get_shape()
        x = atrous_conv_layer(x, [kernel, kernel, in_shape[-1], num_channel], num_channel, 2, name='conv_1')
        x = mygn(x, name='gn_1')
        x = myrelu(x, name='relu_1')
        in_shape = x.get_shape()
        x = atrous_conv_layer(x, [kernel, kernel, in_shape[-1], num_channel], num_channel, 2, name='conv_2')
        x = mygn(x, name='gn_2')
        # Merge
        x = x + shortcut
        x = myrelu(x, name='relu_2')
    return x

def atrous_spatial_pyramid_pooling_block(x, is_train, depth=256, name = 'aspp'):
    in_shape = x.get_shape()
    input_size = tf.shape(x)[1:3]
    filters = [1, 4, 3, 3, 1, 1]
    atrous_rates = [1, 6, 12, 18, 1, 1]
    with tf.variable_scope(name) as scope:
        print('\tBuilding aspp unit: %s' % scope.name)
        # Branch 0: 1x1 conv
        branch0 = conv_layer(x, [filters[0], filters[0], in_shape[3], depth], name="branch0")
        branch0 = mygn(branch0, name='gn_0')
        # Branch 1: 3x3 atrous_conv (rate = 6)
        branch1 = atrous_conv_layer(x, [filters[1], filters[1], in_shape[-1], depth], depth, atrous_rates[1], name='branch1')
        branch1 = mygn(branch1, name='gn_1')
        # Branch 2: 3x3 atrous_conv (rate = 12)
        branch2 = atrous_conv_layer(x, [filters[2], filters[2], in_shape[-1], depth], depth, atrous_rates[2], name='branch2')
        branch2 = mygn(branch2, name='gn_2')
        # Branch 3: 3x3 atrous_conv (rate = 18)
        branch3 = atrous_conv_layer(x, [filters[3], filters[3], in_shape[-1], depth], depth, atrous_rates[3], name='branch3')
        branch3 = mygn(branch3, name='gn_3')
        # Branch 4: image pooling
        # 4.1 global average pooling
        branch4 = tf.reduce_mean(x, [1, 2], name='global_average_pooling', keepdims=True)
        # 4.2 1x1 convolution with 256 filters and batch normalization
        branch4 = conv_layer(x, [filters[4], filters[4], in_shape[3], depth], name="brach4")
        branch4 = mygn(branch4, name='gn_4')
        # 4.3 bilinearly upsample features
        branch4 = tf.image.resize_bilinear(branch4, input_size, name='branch4_upsample')
        # Output
        out = tf.concat([branch0, branch1, branch2, branch3, branch4], axis=3, name='aspp_concat')
        out = myrelu(out, name='relu_out')
        in_shape = out.get_shape()
        out = conv_layer(out, [filters[5], filters[5], in_shape[3], depth], name="aspp_out", relu='no')
        return out

# MODEL ******************************************************************************************************

def deeplab_net(X, classes, is_train=tf.constant(True), pkeep=1):  # ResNet 18

    if pkeep != 1:
        print('\nBuilding train model')
    else:
        print('\nBuilding validation model')

    # filters = [128, 128, 256, 512, 1024]
    filters = [64, 64, 128, 256, 512]
    kernels = [7, 3, 3, 3, 3]
    strides = [2, 0, 2, 2, 2]

    size = tf.shape(X)
    height = size[1]
    width = size[2]

    with tf.variable_scope('generator'):

        # conv1
        print('\tBuilding unit: conv1')
        with tf.variable_scope('conv1'):
            in_shape = X.get_shape()
            x = conv_layer(X, [kernels[0], kernels[0], in_shape[3], filters[0]], name="conv")
            x = mygn(x, name="gn")
            x = myrelu(x)
            x = tf.nn.max_pool(x, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME')

        # conv2_x
        x = residual_block(x, kernels[1], is_train, name='conv2_1')
        x = residual_block(x, kernels[1], is_train, name='conv2_2')

        # conv3_x
        x = first_residual_block(x, kernels[2], filters[2], strides[2], is_train, name='conv3_1')
        x = residual_block(x, kernels[2], is_train, name='conv3_2')

        # conv4_x
        x = first_residual_block(x, kernels[3], filters[3], strides[3], is_train, name='conv4_1')
        x = residual_block(x, kernels[3], is_train, name='conv4_2')

        # conv5_x
        x = first_residual_atrous_block(x, kernels[4], filters[4], strides[4], is_train, name='conv5_1')
        x = residual_atrous_block(x, kernels[4], is_train, name='conv5_2')

        # aspp
        x = atrous_spatial_pyramid_pooling_block(x, is_train, depth=256, name='aspp_1')

        print('\tBuilding unit: class scores')  # Maybe another layer ???
        in_shape = x.get_shape()
        x = conv_layer(x, [1, 1, in_shape[3], classes], name="class_scores")

        # upsample logits
        print('\tBuilding unit: upsample')
        logits = tf.image.resize_images(
            x,
            tf.stack([height, width]),
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

    return logits

#X = tf.ones([1, 256, 256, 3], dtype=tf.float32)
#build_model(X, is_train=tf.constant(True), pkeep=1)