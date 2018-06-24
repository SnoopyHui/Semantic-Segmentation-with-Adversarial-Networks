
import sys
#sys.path.append('/gan_segmentation/')
import layers
import tensorflow as tf

def msca_front(x, weights, pkeep, classes, pretrained = 0):

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

        print('Building generator front model for multi scale context aggregation')
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

        x = tf.transpose(x, [0, 2, 3, 1])

        print('\tBuilding unit: conv5')
        # Convolution 11 - dilated x 2
        x = layers.atrous_conv_layer(x, [3, 3, 512, 512], output_stride = 2, name='conv5_1',
                                     weights=pretrained_vars[10], pkeep=pkeep)
        # Convolution 12 - dilated x 2
        x = layers.atrous_conv_layer(x, [3, 3, 512, 512], output_stride=2, name='conv5_2',
                                     weights=pretrained_vars[11], pkeep=pkeep)

        # Convolution 13 - dilated x 2
        x = layers.atrous_conv_layer(x, [3, 3, 512, 512], output_stride=2, name='conv5_3',
                                     weights=pretrained_vars[12], pkeep=pkeep)

        print('\tBuilding unit: fully conv')
        # Dense-Conv 1 - dilated x 4
        x = layers.atrous_conv_layer(x, [7, 7, 512, 4096], output_stride=4, name='fc6',
                                     weights=pretrained_vars[13], pkeep=pkeep)
        # Dense-Conv 2 - dilated x 4
        x = layers.atrous_conv_layer(x, [1, 1, 4096, 4096], output_stride=4, name='fc7',
                                     weights=pretrained_vars[14], pkeep=pkeep)
        # Final conv
        x = layers.conv_layer(x, [1, 1, 4096, classes], name="final", relu='no')

        # Upsample
        size = tf.shape(x)
        height = size[1]
        width = size[2]
        logits = tf.image.resize_images(
            x,
            tf.stack([height * 8, width * 8]),
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

        return logits

def msca_global(x, weights, pkeep, num_classes, pretrained=0):

    if pretrained == 0:
        pretrained_vars = [0] * 15
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

        print('Building generator global model for multi scale context aggregation')
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

        x = tf.transpose(x, [0, 2, 3, 1])

        print('\tBuilding unit: conv5')
        # Convolution 11 - dilated x 2
        x = layers.atrous_conv_layer(x, [3, 3, 512, 512], output_stride=2, name='conv5_1',
                                     weights=pretrained_vars[10], pkeep=pkeep)
        # Convolution 12 - dilated x 2
        x = layers.atrous_conv_layer(x, [3, 3, 512, 512], output_stride=2, name='conv5_2',
                                     weights=pretrained_vars[11], pkeep=pkeep)

        # Convolution 13 - dilated x 2
        x = layers.atrous_conv_layer(x, [3, 3, 512, 512], output_stride=2, name='conv5_3',
                                     weights=pretrained_vars[12], pkeep=pkeep)

        print('\tBuilding unit: fully conv')
        # Dense-Conv 1 - dilated x 4
        x = layers.atrous_conv_layer(x, [7, 7, 512, 4096], output_stride=4, name='fc6',
                                     weights=pretrained_vars[13], pkeep=pkeep)
        # Dense-Conv 2 - dilated x 4
        x = layers.atrous_conv_layer(x, [1, 1, 4096, 4096], output_stride=4, name='fc7',
                                     weights=pretrained_vars[14], pkeep=pkeep)
        # Final conv
        x = layers.conv_layer(x, [1, 1, 4096, num_classes], name="final", relu='yes')

        with tf.device('/cpu:0'):
            print('Building Context module')
            print('\tBuilding Context units 1-8')

            # Context-Layer 1
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            x = layers.atrous_conv_layer(x, [3, 3, num_classes, num_classes], name = 'context1', output_stride=1, identity_init='yes')
            # Context-Layer 2
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            x = layers.atrous_conv_layer(x, [3, 3, num_classes, num_classes], name = 'context2', output_stride=1, identity_init='yes')
            # Context-Layer 3
            x = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]], mode='CONSTANT')
            x = layers.atrous_conv_layer(x, [3, 3, num_classes, num_classes], name = 'context3', output_stride=2, identity_init='yes')
            # Context-Layer 4
            x = tf.pad(x, [[0, 0], [4, 4], [4, 4], [0, 0]], mode='CONSTANT')
            x = layers.atrous_conv_layer(x, [3, 3, num_classes, num_classes], name = 'context4', output_stride=4, identity_init='yes')
            # Context-Layer 5
            x = tf.pad(x, [[0, 0], [8, 8], [8, 8], [0, 0]], mode='CONSTANT')
            x = layers.atrous_conv_layer(x, [3, 3, num_classes, num_classes], name = 'context5', output_stride=8, identity_init='yes')
            # Context-Layer 6
            x = tf.pad(x, [[0, 0], [16, 16], [16, 16], [0, 0]], mode='CONSTANT')
            x = layers.atrous_conv_layer(x, [3, 3, num_classes, num_classes], name = 'context6', output_stride=16, identity_init='yes')
            # Context-Layer 7
            x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
            x = layers.atrous_conv_layer(x, [3, 3, num_classes, num_classes], name = 'context7', output_stride=1, identity_init='yes')
            # Context-Layer 8
            x = layers.conv_layer(x, [1, 1, num_classes, num_classes], name='global_final', relu='no')

        # Upsample
        size = tf.shape(x)
        height = size[1]
        width = size[2]
        logits = tf.image.resize_images(
            x,
            tf.stack([height * 8, width * 8]),
            method=tf.image.ResizeMethod.BILINEAR,
            align_corners=False)

        return logits


#X = tf.ones([1, 256, 256, 3], dtype=tf.float32)
#msca_global(X, weights=0, pkeep=1, num_classes=8, pretrained=0)