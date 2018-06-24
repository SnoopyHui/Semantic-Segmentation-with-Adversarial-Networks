
import sys
from layers import WEIGHT_DECAY_KEY
sys.path.append('/gan_segmentation/models/discriminators')
import largeFOV, smallFOV, stanford_background_dataset_discriminator
sys.path.append('/gan_segmentation/models/generators')
import fcn32, unet, deeplab_v3
import tensorflow as tf
import numpy as np
from utils import compute_accuracy
import pdb

class GAN_trainer:

    def __init__(self, x, y, num_classes, generator = 'fcn32', discriminator = 'smallFOV', optimizer = 'adagrad', is_train = 1):

        self.x = x
        self.y = y
        self.num_classes = num_classes
        self.generator = generator
        self.discriminator = discriminator
        self.optimizer = optimizer
        self.loss_bce_weight = tf.placeholder(tf.float32)
        self.pkeep = tf.placeholder(tf.float32)
        self.lr_g = tf.placeholder(tf.float32)
        self.lr_d = tf.placeholder(tf.float32)
        self.weight_decay = tf.placeholder(tf.float32)
        if is_train == 1:
            self.is_train = tf.constant(True)
        else:
            self.is_train = tf.constant(False)

    def make_dis_input(self, G_logits, image, label):

        variant = "scaling" # basic, product or scaling
        tau = tf.constant(0.9, dtype=tf.float32)

        # 0. down-sample labels and image
        logits_shape = tf.shape(G_logits)
        downsampling_shape = [logits_shape[1], logits_shape[2]]
        label = tf.image.resize_images(label, size = downsampling_shape)
        label = tf.cast(label, dtype=tf.int32)
        image = tf.image.resize_images(image, size = downsampling_shape)

        # 1. one hot representation of labels
        G_probs = tf.nn.softmax(G_logits, name='softmax_tensor')
        batch = tf.cast(tf.shape(label)[0], dtype=tf.int32)
        height = tf.cast(tf.shape(label)[1], dtype=tf.int32)
        width = tf.cast(tf.shape(label)[2], dtype=tf.int32)
        one_hot_flat_y = tf.one_hot(tf.reshape(label, [-1, ]), self.num_classes, axis=1)
        one_hot_y = tf.reshape(one_hot_flat_y,[batch, height, width, self.num_classes])

        if variant == "basic":

            # define operations between generator and discriminator - version "basic"

            self.c_prime = self.num_classes
            fake_disciminator_input = G_probs
            real_disciminator_input = one_hot_y
            return real_disciminator_input, fake_disciminator_input

        elif variant == "product":

            # define operations between generator and discriminator - version "product"

            self.c_prime = self.num_classes*3

            # 2. Slice r,g,b components
            blue = tf.slice(image, [0,0,0,0], [1, height, width, 1])
            green = tf.slice(image, [0, 0, 0, 1], [1, height, width, 1])
            red = tf.slice(image, [0, 0, 0, 2], [1, height, width, 1])

            # 3. Generate fake discriminator input
            product_b = G_probs * blue
            product_g = G_probs * green
            product_r = G_probs * red
            fake_disciminator_input = tf.concat([product_b, product_g, product_r], axis=3)

            # 4. Generate also real discriminator input
            product_b = one_hot_y * blue
            product_g = one_hot_y * green
            product_r = one_hot_y * red
            real_disciminator_input = tf.concat([product_b, product_g, product_r], axis=3)

            return real_disciminator_input, fake_disciminator_input

        elif variant == "scaling":

            # define operations between generator and discriminator - version "scaling"

            self.c_prime = self.num_classes

            fake_disciminator_input = G_probs

            #2. replace labels

            yil = tf.reduce_sum(tf.where(tf.greater(one_hot_y, 0.),
                                         tf.maximum(G_probs, tau),
                                         tf.zeros_like(one_hot_y)), axis = 3)
            sil = tf.reduce_sum(tf.where(tf.equal(one_hot_y, tf.constant(1, dtype=tf.float32)),
                                         G_probs,
                                         tf.zeros_like(one_hot_y)), axis=3)

            yil = tf.expand_dims(yil, axis=3)
            sil = tf.expand_dims(sil, axis=3)
            yil = tf.concat([yil]*self.num_classes, axis=3)
            sil = tf.concat([sil]*self.num_classes, axis=3)

            real_disciminator_input = tf.where(tf.equal(one_hot_y, 1.),
                                               yil,
                                               G_probs*((1-yil)/(1-sil)))

            return real_disciminator_input, fake_disciminator_input

    def get_loss_discriminator(self, real_D_logits, fake_D_logits):

        # Define logits batch_size, height, width
        in_shape = tf.shape(fake_D_logits)
        batch_size = tf.cast(in_shape[0], dtype=tf.int32)
        height = tf.cast(in_shape[1], dtype=tf.int32)
        width = tf.cast(in_shape[2], dtype=tf.int32)

        # Reshape logits by num of classes
        fake_D_logits_by_num_classes = tf.reshape(fake_D_logits, [-1, 2])
        real_D_logits_by_num_classes = tf.reshape(real_D_logits, [-1, 2])

        # Define real/fake labels
        label_real = tf.cast(tf.fill([batch_size*height*width], 1.0), dtype=tf.int32)
        label_fake = tf.cast(tf.fill([batch_size*height*width], 0.0), dtype=tf.int32)

        # Compute loss
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=real_D_logits_by_num_classes,
                                                                              labels=label_real,
                                                                              name="bce_1"))
        loss += tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fake_D_logits_by_num_classes,
                                                                              labels=label_fake,
                                                                              name="bce_2"))
        # Add l2 cost
        costs = [tf.nn.l2_loss(var) for var in tf.get_collection(WEIGHT_DECAY_KEY, scope="model/discriminator")]
        l2_loss = tf.multiply(self.weight_decay, tf.add_n(costs))
        total_loss = loss + l2_loss

        return total_loss

    def get_loss_generator(self, labels, G_logits, fake_D_logits):

        # Find valid indices
        logits_by_num_classes = tf.reshape(G_logits, [-1, self.num_classes])
        preds = tf.argmax(G_logits, axis=3, output_type=tf.int32)
        preds_flat = tf.reshape(preds, [-1, ])
        labels_flat = tf.reshape(labels, [-1, ])
        valid_indices = tf.multiply(tf.to_int32(labels_flat <= self.num_classes - 1), tf.to_int32(labels_flat > -1))

        # Prepare segmentation model logits and labels
        valid_logits = tf.dynamic_partition(logits_by_num_classes, valid_indices, num_partitions=2)[1]
        valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions=2)[1]
        valid_preds = tf.dynamic_partition(preds_flat, valid_indices, num_partitions=2)[1]

        in_shape = tf.shape(fake_D_logits)
        batch_size = tf.cast(in_shape[0], dtype=tf.int32)
        height = tf.cast(in_shape[1], dtype=tf.int32)
        width = tf.cast(in_shape[2], dtype=tf.int32)

        fake_D_logits_by_num_classes = tf.reshape(fake_D_logits, [-1, 2])
        label_real = tf.cast(tf.fill([batch_size*height*width], 1.0), dtype=tf.int32)

        l_mce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=valid_logits,
                                                               labels=valid_labels,
                                                               name="g_mce"))
        l_bce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fake_D_logits_by_num_classes,
                                                                              labels=label_real,
                                                                              name="l_bce"))
        loss = l_mce + self.loss_bce_weight * l_bce

        # Add l2 loss
        costs = [tf.nn.l2_loss(var) for var in tf.get_collection(WEIGHT_DECAY_KEY, scope="model/generator")]
        l2_loss = tf.multiply(self.weight_decay, tf.add_n(costs))
        total_loss = loss + l2_loss

        return total_loss, valid_logits, valid_labels, valid_preds

    def train_op(self):

        image = self.x
        labels = self.y

        # Define generator
        if self.generator == 'fcn32':
            #G_logits = fcn32.fcn32(image, self.pkeep, self.num_classes)
            G_unscaled, G_logits = fcn32.fcn32(image, self.pkeep, self.num_classes)
        elif self.generator == 'fcn32_DCGAN':
            G_logits = fcn32_DCGAN.fcn32(image, self.pkeep, self.num_classes)
        elif self.generator == 'unet':
            G_logits = unet.unet(image, self.pkeep, self.num_classes, channels=3, num_layers=5)
        elif self.generator == 'deeplab_v3':
            G_logits = deeplab_v3.deeplab_net(image, self.num_classes, is_train=self.is_train, pkeep=self.pkeep)
        else:
            print('error! specified generator is not valid')
            sys.exit(1)

        # Define discriminators input
        #labels, real_disciminator_input, fake_disciminator_input = self.make_dis_input(G_logits, image, labels)
        real_disciminator_input, fake_disciminator_input = self.make_dis_input(G_unscaled, image, labels)

        # Define 2 discriminators: fake and real
        if self.discriminator == 'smallFOV':
            print('Building discriminator smallFOV')
            with tf.variable_scope('discriminator'):
                real_D_logits = smallFOV.smallFOV(real_disciminator_input, c_prime=self.c_prime)
            with tf.variable_scope('discriminator', reuse=True):
                fake_D_logits = smallFOV.smallFOV(fake_disciminator_input, c_prime=self.c_prime)
        elif self.discriminator == 'smallFOV_DCGAN':
            print('Building discriminator smallFOV_DCGAN')
            with tf.variable_scope('discriminator'):
                real_D_logits = smallFOV_DCGAN.smallFOV(real_disciminator_input, c_prime=self.c_prime)
            with tf.variable_scope('discriminator', reuse=True):
                fake_D_logits = smallFOV_DCGAN.smallFOV(fake_disciminator_input, c_prime=self.c_prime)
        elif self.discriminator == 'largeFOV':
            print('Building discriminator largeFOV')
            with tf.variable_scope('discriminator'):
                real_D_logits = largeFOV.largeFOV(real_disciminator_input, c_prime=self.c_prime)
            with tf.variable_scope('discriminator', reuse=True):
                fake_D_logits = largeFOV.largeFOV(fake_disciminator_input, c_prime=self.c_prime)
        elif self.discriminator == 'largeFOV_DCGAN':
            print('Building discriminator largeFOV_DCGAN')
            with tf.variable_scope('discriminator'):
                real_D_logits = largeFOV_DCGAN.largeFOV(real_disciminator_input, c_prime=self.c_prime)
            with tf.variable_scope('discriminator', reuse=True):
                fake_D_logits = largeFOV_DCGAN.largeFOV(fake_disciminator_input, c_prime=self.c_prime)
        elif self.discriminator == 'sbd':
            print('Building discriminator SBD')
            with tf.variable_scope('discriminator'):
                real_D_logits = stanford_background_dataset_discriminator.stanford_bd_model(image,
                                                                                            real_disciminator_input,
                                                                                            c_prime=self.c_prime)
            with tf.variable_scope('discriminator', reuse=True):
                fake_D_logits = stanford_background_dataset_discriminator.stanford_bd_model(image,
                                                                                            fake_disciminator_input,
                                                                                            c_prime=self.c_prime)
        else:
            print('error! specified discriminator is not valid')
            sys.exit(1)

        # Define losses
        D_loss = self.get_loss_discriminator(real_D_logits, fake_D_logits)
        G_loss, valid_logits, valid_labels, valid_preds = self.get_loss_generator(labels, G_logits, fake_D_logits)

        # Improved wasserstein gan penalty
        #epsilon = tf.random_uniform([], 0.0, 1.0)
        #x_hat = real_disciminator_input * epsilon + (1 - epsilon) * fake_disciminator_input
        #with tf.variable_scope('discriminator', reuse=True):
        #    d_hat = largeFOV.largeFOV(x_hat, c_prime=self.c_prime)
        #gradients = tf.gradients(d_hat, x_hat)[0]
        #slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
        #gradient_penalty = 10 * tf.reduce_mean((slopes - 1.0) ** 2)
        #D_loss += gradient_penalty

        # Define segmentation accuracy
        pixel_acc, mean_iou_acc, mean_per_class_acc = compute_accuracy(valid_preds, valid_labels, self.num_classes, 'accuracy')

        # Define optimizers
        if self.optimizer == 'adagrad':
            D_optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr_d)
            G_optimizer = tf.train.AdagradOptimizer(learning_rate=self.lr_g)
        elif self.optimizer == 'sgd':
            D_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr_d)
            G_optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr_g)
        elif self.optimizer == 'adam':
            D_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_d, epsilon=1e-05)
            G_optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_g, epsilon=1e-05)

        D_grads = D_optimizer.compute_gradients(D_loss, var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/discriminator'))
        G_grads = G_optimizer.compute_gradients(G_loss, var_list= tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='model/generator'))
        D_train_step = D_optimizer.apply_gradients(D_grads)
        G_train_step = G_optimizer.apply_gradients(G_grads)

        return G_train_step, D_train_step, G_loss, D_loss, pixel_acc, mean_iou_acc, mean_per_class_acc

