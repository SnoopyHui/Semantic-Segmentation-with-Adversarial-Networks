
# Imports
from __future__ import print_function, division
import tensorflow as tf
import warnings
from tensorflow.python.ops import control_flow_ops
import numpy as np
import pdb

# DATA AUGMENTATION ****************************************************************************************************

def random_rotation_image_with_annotation(image_tensor, annotation_tensor, max_angle):

    # Random variable: two possible outcomes (0 or 1)
    # with 0.5 chance
    random_var = tf.cast(tf.random_uniform(maxval=2, dtype=tf.int32, shape=[]),dtype=tf.float32)

    # Random selection of angle and direction of rotation
    random_angle = tf.cast(tf.random_uniform(maxval=max_angle, dtype=tf.int32, shape=[]),dtype=tf.float32)
    random_direction = tf.cast(tf.random_uniform(minval=-1, maxval=1, dtype=tf.int32, shape=[]),dtype=tf.float32)
    randomly_rotated_img = control_flow_ops.cond(pred=tf.equal(tf.multiply(tf.abs(random_direction), random_var), 0),
                                                 true_fn=lambda: tf.contrib.image.rotate(image_tensor,
                                                                                         random_direction * random_angle,
                                                                                         interpolation='NEAREST'),
                                                 false_fn=lambda: image_tensor)
    randomly_rotated_annotation = control_flow_ops.cond(pred=tf.equal(tf.multiply(tf.abs(random_direction), random_var), 0),
                                                 true_fn=lambda: tf.contrib.image.rotate(annotation_tensor,
                                                                                         random_direction * random_angle,
                                                                                         interpolation='NEAREST'),
                                                 false_fn=lambda: annotation_tensor)

    return randomly_rotated_img, randomly_rotated_annotation

def flip_randomly_left_right_image_with_annotation(image_tensor, annotation_tensor):
    """Accepts image tensor and annotation tensor and returns randomly flipped tensors of both.
    The function performs random flip of image and annotation tensors with probability of 1/2
    The flip is performed or not performed for image and annotation consistently, so that
    annotation matches the image.

    Parameters
    ----------
    image_tensor : Tensor of size (width, height, 3)
        Tensor with image
    annotation_tensor : Tensor of size (width, height, 1)
        Tensor with annotation

    Returns
    -------
    randomly_flipped_img : Tensor of size (width, height, 3) of type tf.float.
        Randomly flipped image tensor
    randomly_flipped_annotation : Tensor of size (width, height, 1)
        Randomly flipped annotation tensor

    """

    # Random variable: two possible outcomes (0 or 1)
    # with 0.5 chance
    random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])


    randomly_flipped_img = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                 true_fn=lambda: tf.image.flip_left_right(image_tensor),
                                                 false_fn=lambda: image_tensor)

    randomly_flipped_annotation = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                        true_fn=lambda: tf.image.flip_left_right(annotation_tensor),
                                                        false_fn=lambda: annotation_tensor)

    return randomly_flipped_img, randomly_flipped_annotation

def flip_randomly_up_down_image_with_annotation(image_tensor, annotation_tensor):
    """Accepts image tensor and annotation tensor and returns randomly flipped tensors of both.
       The function performs random flip of image and annotation tensors with probability of 1/2
       The flip is performed or not performed for image and annotation consistently, so that
       annotation matches the image.

       Parameters
       ----------
       image_tensor : Tensor of size (width, height, 3)
           Tensor with image
       annotation_tensor : Tensor of size (width, height, 1)
           Tensor with annotation

       Returns
       -------
       randomly_flipped_img : Tensor of size (width, height, 3) of type tf.float.
           Randomly flipped image tensor
       randomly_flipped_annotation : Tensor of size (width, height, 1)
           Randomly flipped annotation tensor

       """

    # Random variable: two possible outcomes (0 or 1)
    # with 0.5 chance
    random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])

    randomly_flipped_img = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                 true_fn=lambda: tf.image.flip_up_down(image_tensor),
                                                 false_fn=lambda: image_tensor)

    randomly_flipped_annotation = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                        true_fn=lambda: tf.image.flip_up_down(annotation_tensor),
                                                        false_fn=lambda: annotation_tensor)

    return randomly_flipped_img, randomly_flipped_annotation

def random_color_distortion(image_tensor, annotation_tensor):

    random_var_brightness = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    distorted_image = control_flow_ops.cond(pred=tf.equal(random_var_brightness, 0),
                                                true_fn=lambda: tf.image.random_brightness(image_tensor, max_delta=32. / 255.),
                                                false_fn=lambda: image_tensor)
    random_var_saturation = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    distorted_image = control_flow_ops.cond(pred=tf.equal(random_var_saturation, 0),
                                            true_fn=lambda: tf.image.random_saturation(distorted_image, lower=0.5, upper=1.5),
                                            false_fn=lambda: distorted_image)
    random_var_hue = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    distorted_image = control_flow_ops.cond(pred=tf.equal(random_var_hue, 0),
                                            true_fn=lambda: tf.image.random_hue(distorted_image, max_delta=0.2),


                                            false_fn=lambda: distorted_image)
    random_var_contrast = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])
    distorted_image = control_flow_ops.cond(pred=tf.equal(random_var_contrast, 0),
                                            true_fn=lambda: tf.image.random_contrast(distorted_image, lower=0.5, upper=1.5),
                                            false_fn=lambda: distorted_image)

    return tf.clip_by_value(distorted_image, 0.0, 1.0), annotation_tensor

# ACCURACY FUNCTIONS ***************************************************************************************************

def compute_accuracy(valid_preds, valid_labels, classes , name = 'accuracy'):
    with tf.name_scope(name):
        #pixel_acc = tf.divide(tf.reduce_sum(tf.cast(tf.equal(valid_labels, valid_preds), dtype=tf.int32)),
        #                      tf.cast(tf.shape(valid_labels)[0], dtype=tf.int32))
        _, pixel_acc = tf.metrics.accuracy(valid_labels, valid_preds)
        #cm = tf.confusion_matrix(valid_labels, valid_preds, num_classes=CLASSES)
        _, cm = tf.metrics.mean_iou(valid_labels, valid_preds, classes)
        mean_iou = compute_mean_iou(cm)
        _, mean_per_class_acc = tf.metrics.mean_per_class_accuracy(valid_labels, valid_preds, classes)
    return pixel_acc, mean_iou, mean_per_class_acc

def compute_mean_iou(total_cm, name='mean_iou'):
    """Compute the mean intersection-over-union via the confusion matrix."""
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)

    return result

# DECAY FUNCTIONS ******************************************************************************************************

def lr_decay(learning_rate):
    return (learning_rate * 0.5)

# NORMALIZATIONS *******************************************************************************************************

def mybn(x, is_train, name='bn'):
    moving_average_decay = 0.9
    with tf.variable_scope(name):
        decay = moving_average_decay
        # Get batch mean and var, which will be used during training
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
        # Define variables, mu and sigma are not trainable since depend on the batch (train) or the population (test)
        with tf.device('/CPU:0'):
            mu = tf.get_variable('mu', batch_mean.get_shape(), tf.float32,
                                 initializer=tf.zeros_initializer(), trainable=False)
            sigma = tf.get_variable('sigma', batch_var.get_shape(), tf.float32,
                                    initializer=tf.ones_initializer(), trainable=False)
            beta = tf.get_variable('beta', batch_mean.get_shape(), tf.float32,
                                   initializer=tf.zeros_initializer())
            gamma = tf.get_variable('gamma', batch_var.get_shape(), tf.float32,
                                    initializer=tf.ones_initializer())
        update = 1.0 - decay
        update_mu = mu.assign_sub(update * (mu - batch_mean))
        update_sigma = sigma.assign_sub(update * (sigma - batch_var))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_mu)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_sigma)

        mean, var = tf.cond(is_train, lambda: (batch_mean, batch_var),
                            lambda: (mu, sigma))
        bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
    return bn

def mygn(x, G=32, eps=1e-5, name='gn'):
    with tf.variable_scope(name):
        # NHWC to NCHW
        #x = tf.transpose(x, [0, 3, 1, 2])

        _, channels, _, _ = x.get_shape().as_list()

        shape = tf.shape(x)
        N = shape[0]
        C = shape[1]
        H = shape[2]
        W = shape[3]

        x = tf.reshape(x, [N, G, C//G, H, W])

        group_mean, group_var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - group_mean) / tf.sqrt(group_var + eps)

        with tf.device('/CPU:0'):
            beta = tf.get_variable('beta', [channels], initializer=tf.constant_initializer(0.0))
            gamma = tf.get_variable('gamma', [channels], initializer=tf.constant_initializer(1.0))

        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        x = tf.reshape(x, [N, C, H, W]) * gamma + beta

        # NCHW to NHWC
        #x = tf.transpose(x, [0, 2, 3, 1])

    return x

# RELU *****************************************************************************************************************

def myrelu(x, leakness=0.0, name=None):
    if leakness > 0.0:
        name = 'lrelu' if name is None else name
        return tf.maximum(x, x*leakness, name='lrelu')
    else:
        name = 'relu' if name is None else name
        return tf.nn.relu(x, name='relu')

# UNET OUTPUT SIZE *****************************************************************************************************

def compute_unet_output_size(in_size, num_layers):
    size = in_size
    reduction_due_to_conv = 2
    for i in range(num_layers):
        size -= 2*2*reduction_due_to_conv # 2 convolutions, 2 time in every layer
        reduction_due_to_conv *= 2
    size += reduction_due_to_conv # bottom layer only visited once
    return size

def is_valid_input_unet(in_size, num_layers):
    isvalid = 1
    size = in_size
    reduction_due_to_conv = 2
    for i in range(num_layers-1):
        size -= reduction_due_to_conv*2 # 2 convolutions, 2 time in every layer
        if size % 2 != 0:
            #print('Error: odd image size before pooling at layer ' + str(num_layers-i) + ', ' + str(size))
            isvalid = 0
        size  = size / 2
    return isvalid

# INITIALIZER FUNCTIONS ************************************************************************************************

def identity_initializer(filter_shape):

    """returns the values of a filter that simply passes forward the input feature map"""

    filter = np.zeros((filter_shape))
    center = int(filter_shape[1]/2)
    for i in range(filter_shape[2]):
            filter[center, center, i, i] = np.float(1)
    return filter

# SPECTRAL NORMED WEIGHTS

NO_OPS = 'NO_OPS'


def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
  # Usually num_iters = 1 will be enough
  W_shape = W.shape.as_list()
  W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
  if u is None:
    u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
  def power_iteration(i, u_i, v_i):
    v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
    u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
    return i + 1, u_ip1, v_ip1
  _, u_final, v_final = tf.while_loop(
    cond=lambda i, _1, _2: i < num_iters,
    body=power_iteration,
    loop_vars=(tf.constant(0, dtype=tf.int32),
               u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
  )
  if update_collection is None:
    warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                  '. Please consider using a update collection instead.')
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    with tf.control_dependencies([u.assign(u_final)]):
      W_bar = tf.reshape(W_bar, W_shape)
  else:
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)
    # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
    # has already been collected on the first call.
    if update_collection != NO_OPS:
      tf.add_to_collection(update_collection, u.assign(u_final))
  if with_sigma:
    return W_bar, sigma
  else:
    return W_bar