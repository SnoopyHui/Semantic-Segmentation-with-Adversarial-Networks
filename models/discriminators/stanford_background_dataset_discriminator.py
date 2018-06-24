
import sys
sys.path.append('/Users/albertbou/PycharmProjects/AdversarialSemanticSegmentation')
import layers
import tensorflow as tf
import pdb

def stanford_bd_model(image, segmentation, c_prime):

    # NHWC TO NCHW *****************************************************************************************************

    image = tf.transpose(image, [0, 3, 1, 2])
    segmentation = tf.transpose(segmentation, [0, 3, 1, 2])

    # DEFINE MODEL *****************************************************************************************************

    # Branch 1
    # Convolution 1
    b1 = layers.conv_layer(segmentation, [5, 5, c_prime, 64], name = "b1_conv1", data_format='NCHW')

    # Branch 2
    # Convolution 1
    b2 = layers.conv_layer(image, [5, 5, 3, 16], name="b2_conv1", data_format='NCHW')
    # Convolution 2
    b2 = layers.conv_layer(b2, [5, 5, 16, 64], name="b2_conv2", data_format='NCHW')

    # Feature concatenation
    feat_concat = tf.concat([b1, b2], axis = 1)

    # Merged branch
    # Convolution 1
    x = layers.conv_layer(feat_concat, [3, 3, 128, 128], name="m_conv1", data_format='NCHW')
    # Max-Pooling 1
    x = layers.max_pool_layer(x, padding='SAME', data_format='NCHW')
    # Convolution 2
    x = layers.conv_layer(x, [3, 3, 128, 256], name="m_conv2", data_format='NCHW')
    # Max-Pooling 2
    x = layers.max_pool_layer(x, padding='SAME', data_format='NCHW')
    # Convolution 3
    x = layers.conv_layer(x, [3, 3, 256, 512], name="m_conv3", data_format='NCHW')
    # Convolution 4
    x = layers.conv_layer(x, [3, 3, 512, 2], name="m_conv4", data_format='NCHW', relu='no')
    # Average-Pooling
    x = tf.transpose(x, [0, 2, 3, 1])
    #x = tf.reduce_mean(x, axis = [1,2])
    # Reshape
    #x = tf.reshape(x, (1, 2))
    return x
