import sys
sys.path.append('/Users/albertbou/PycharmProjects/AdversarialSemanticSegmentation')
from utils import myrelu, mygn
import layers
import tensorflow as tf

def smallFOV(x, c_prime):

    # NHWC TO NCHW *****************************************************************************************************

    x = tf.transpose(x, [0, 3, 1, 2])

    # DEFINE MODEL *****************************************************************************************************

    # Convolution 1
    x = layers.conv_layer(x, [3, 3, c_prime, 96], name = "d_conv1", data_format='NCHW')
    # Convolution 2
    x = layers.conv_layer(x, [1, 1, 96, 128], name="d_conv2", data_format='NCHW')
    # Max-Pooling 1
    x = layers.avg_pool_layer(x, padding='SAME', data_format='NCHW')
    # Convolution 3
    x = layers.conv_layer(x, [3, 3, 128, 256], name="d_conv3", data_format='NCHW')
    # Convolution 4
    x = layers.conv_layer(x, [1, 1, 256, 256], name="d_conv4", data_format='NCHW')
    # Max-Pooling 2
    x = layers.avg_pool_layer(x, padding='SAME', data_format='NCHW')
    # Convolution 5
    x = layers.conv_layer(x, [3, 3, 256, 512], name="d_conv5", data_format='NCHW')
    # Convolution 6
    x = layers.conv_layer(x, [1, 1, 512, 2], name="d_conv6", data_format='NCHW', relu='no')

    # NCHW TO NHWC *****************************************************************************************************

    x = tf.transpose(x, [0, 2, 3, 1])

    return x