
import tensorflow as tf
from utils import random_color_distortion, flip_randomly_left_right_image_with_annotation, classes

# FCN32 INPUT PIPELINE *************************************************************************************************

def fcn_parse_fn(example):

    "Parse TFExample records and perform simple data augmentation."

    feature = {'height': tf.FixedLenFeature([], tf.int64),
             'width': tf.FixedLenFeature([], tf.int64),
             'image': tf.FixedLenFeature([], tf.string),
             'label': tf.FixedLenFeature([], tf.string)}


    # Decode
    parsed = tf.parse_single_example(example, feature)
    image = tf.decode_raw(parsed['image'], tf.int8)
    label = tf.decode_raw(parsed['label'], tf.int8)
    height = tf.cast(parsed['height'], tf.int32)
    width = tf.cast(parsed['width'], tf.int32)
    image = tf.reshape(image, tf.stack([height, width, 3]))
    label = tf.reshape(label, tf.stack([height, width, 1]))

    # Random cropping
    combined = tf.concat([image, label], axis=2)
    combined_crop = tf.random_crop(combined, [256, 256, 4])
    image = tf.slice(combined_crop, [0, 0, 0], [256, 256, 3])
    label = tf.slice(combined_crop, [0, 0, 3], [256, 256, 1])
    label = tf.cast(label, dtype=tf.float32)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.divide(image, 255.)

    # Data Augmentation
    image, label = random_color_distortion(image, label)
    image, label = flip_randomly_left_right_image_with_annotation(image, label)
    # image, label = random_rotation_image_with_annotation(image, label, 5) - I decide not to rotate
    # image, label = flip_randomly_up_down_image_with_annotation(image, label) - I decide not to flip vertically

    # Normalize data and labels
    image = tf.subtract(image, 127. / 255.)
    label = tf.cast(label, dtype=tf.int32)

    return image, label

def fcn_parse_fn_val_and_test(example):

    "Parse TFExample records and perform simple data augmentation."

    feature = {'height': tf.FixedLenFeature([], tf.int64),
               'width': tf.FixedLenFeature([], tf.int64),
               'image': tf.FixedLenFeature([], tf.string),
               'label': tf.FixedLenFeature([], tf.string)}

    # Decode
    parsed = tf.parse_single_example(example, feature)
    image = tf.decode_raw(parsed['image'], tf.int8)
    label = tf.decode_raw(parsed['label'], tf.int8)
    height = tf.cast(parsed['height'], tf.int32)
    width = tf.cast(parsed['width'], tf.int32)
    image = tf.reshape(image, tf.stack([height, width, 3]))
    label = tf.reshape(label, tf.stack([height, width, 1]))

    # Random cropping
    combined = tf.concat([image, label], axis=2)
    combined_crop = tf.random_crop(combined, [256, 256, 4])
    image = tf.slice(combined_crop, [0, 0, 0], [256, 256, 3])
    label = tf.slice(combined_crop, [0, 0, 3], [256, 256, 1])
    label = tf.cast(label, dtype=tf.float32)
    image = tf.cast(image, dtype=tf.float32)
    image = tf.divide(image, 255.)

    # Normalize data and labels
    image = tf.subtract(image, 127. / 255.)
    label = tf.cast(label, dtype=tf.int32)

    return image, label

def fcn_read_and_decode(data_path, epochs, batch_size):
    buffer_size = 500 * 600 * 4
    dataset = tf.data.TFRecordDataset(data_path, buffer_size=buffer_size, num_parallel_reads=64)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(8000, epochs))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(fcn_parse_fn, batch_size, num_parallel_batches=2))
    dataset = dataset.prefetch(2)
    iterator = dataset.make_one_shot_iterator()
    batch_images, batch_labels = iterator.get_next()
    return batch_images, batch_labels

def fcn_read_and_decode_val_and_test(data_path, epochs, batch_size = 1):
    buffer_size = 500 * 600 * 4
    dataset = tf.data.TFRecordDataset(data_path, buffer_size=buffer_size, num_parallel_reads=64)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(8000, epochs))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(fcn_parse_fn_val_and_test, 1, num_parallel_batches=2))
    dataset = dataset.prefetch(2)
    iterator = dataset.make_one_shot_iterator()
    batch_images, batch_labels = iterator.get_next()
    return batch_images, batch_labels

# UNET INPUT PIPELINE **************************************************************************************************

def unet_parse_fn(example):

  "Parse TFExample records and perform simple data augmentation."

  feature = {'height': tf.FixedLenFeature([], tf.int64),
             'width': tf.FixedLenFeature([], tf.int64),
             'image': tf.FixedLenFeature([], tf.string),
             'label': tf.FixedLenFeature([], tf.string)}

  parsed = tf.parse_single_example(example, feature)

  image = tf.decode_raw(parsed['image'], tf.float32)
  label = tf.decode_raw(parsed['label'], tf.float32)
  height = tf.cast(parsed['height'], tf.int32)
  width = tf.cast(parsed['width'], tf.int32)
  image = tf.reshape(image, tf.stack([height, width, 3]))
  label = tf.reshape(label, tf.stack([height, width, 1]))
  image = tf.divide(image, 255.)

  # Data Augmentation
  image, label = random_color_distortion(image, label)
  image, label = flip_randomly_left_right_image_with_annotation(image, label)
  #image, label = random_rotation_image_with_annotation(image, label, 5) - I decide not to rotate
  #image, label = flip_randomly_up_down_image_with_annotation(image, label) - I decide not to flip vertically

  # both height and width have to be multiple multiple of (size-valid_size_offset) / valid_size_gap
  target_height = tf.cast((tf.floor(tf.cast(height, dtype=tf.float32)-12/16)*16)+12, dtype=tf.int32)
  target_width = tf.cast((tf.floor(tf.cast(width, dtype=tf.float32)-12/16)*16)+12, dtype=tf.int32)

  image = tf.image.resize_image_with_crop_or_pad(image, target_height, target_width)
  label = tf.image.resize_image_with_crop_or_pad(label, target_height, target_width)

  # Normalize data and labels
  image = tf.subtract(image, 127. / 255.)
  label = tf.cast(label, dtype=tf.int32)

  return image, label

def unet_read_and_decode(data_path, epochs, batch_size):

    buffer_size = 8 * 500 * 600 * 4
    dataset = tf.data.TFRecordDataset(data_path, buffer_size=buffer_size, num_parallel_reads=64)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(10000, epochs))
    dataset = dataset.apply(tf.contrib.data.map_and_batch(unet_parse_fn, batch_size=batch_size, num_parallel_batches=8))
    dataset = dataset.prefetch(8)
    iterator = dataset.make_one_shot_iterator()
    batch_images, batch_labels = iterator.get_next()
    return batch_images, batch_labels
