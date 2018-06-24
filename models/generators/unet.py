
import sys
sys.path.append('/Users/albertbou/PycharmProjects/AdversarialSemanticSegmentation/models')
import layers
from utils import compute_unet_output_size, is_valid_input_unet
from collections import OrderedDict
import tensorflow as tf
import pdb

def unet(x, pkeep, num_classes, channels=3, num_layers=5):

    features_root = 64
    filter_size = 3
    pool_size = 2
    in_node = x
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    # DEFINE MODEL *****************************************************************************************************

    with tf.variable_scope('generator'):
        print('\nBuilding generator model u-net')
        print('\tBuilding encoder')

        # down layers
        for layer in range(0, num_layers):
            features = 2 ** layer * features_root
            print('\t layer '+str(layer)+': ' +str(features)+ ' features channels')
            if layer == 0:
                filter_shape1 = [filter_size, filter_size, channels, features]
                filter_shape2 = [filter_size, filter_size, features, features]
            else:
                filter_shape1 = [filter_size, filter_size, features // 2, features]
                filter_shape2 = [filter_size, filter_size, features, features]

            conv1 = layers.conv_layer(in_node, filter_shape1, pkeep, 'down_conv1_layer' + str(layer), padding='VALID')
            conv2 = layers.conv_layer(conv1, filter_shape2, pkeep, 'down_conv2_layer' + str(layer), padding='VALID')
            dw_h_convs[layer] = conv2

            if layer < num_layers - 1:
                pools[layer] = layers.max_pool_layer(dw_h_convs[layer])
                in_node = pools[layer]
        in_node = dw_h_convs[num_layers - 1]
        print('\tBuilding decoder')
        print('\t layer ' + str(layer) + ': ' + str(features) + ' features channels')

        # up layers
        for layer in range(num_layers - 2, -1, -1):
            features = 2 ** (layer + 1) * features_root

            filter_shape = [pool_size, pool_size, features // 2, features]
            h_deconv = layers.deconv_layer(in_node, filter_shape, 'deconv_layer' + str(layer), padding='VALID')
            h_deconv_concat = layers.crop_and_concat_layer(dw_h_convs[layer], h_deconv)
            deconv[layer] = h_deconv_concat

            filter_shape1 = [filter_size, filter_size, features, features // 2]
            filter_shape2 = [filter_size, filter_size, features // 2, features // 2]
            print('\t layer '+str(layer)+': ' +str(features // 2)+ ' features channels')
            conv1 = layers.conv_layer(h_deconv_concat, filter_shape1, pkeep, 'up_conv1_layer' + str(layer), padding='VALID')
            conv2 = layers.conv_layer(conv1, filter_shape2, pkeep, 'up_conv2_layer' + str(layer), padding='VALID')
            up_h_convs[layer] = conv2
            in_node = up_h_convs[layer]

        # Output Map
        filter_shape = [1, 1, features_root, num_classes]
        output_map = layers.final_layer(in_node, filter_shape, padding='VALID')
        logits = output_map

    return logits

#X = tf.ones((1, 476, 636, 3))
#y = tf.ones((1, 476, 636, 1))
#model = unet(X, pkeep = 1 , is_train = tf.constant(True))

#for i in range (0, 1000):
#    valid_size = is_valid_input_unet(i, num_layers=4)
#    if valid_size == 1:
#        print('Imput size ' + str(i) + ' is valid')
#    else:
#        print('Imput size ' + str(i) + ' is not valid')