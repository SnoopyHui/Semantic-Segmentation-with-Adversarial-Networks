#!/usr/bin/env python

# Imports
from __future__ import print_function, division
import sys
import updater
from utils import *
from input_pipeline import*
from datetime import datetime
from hyperopt import hp, fmin, tpe
from layers import SPECTRAL_NORM_UPDATE_OPS
import os

# Disable Warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define generator (fcn32s/unet/deeplab_v3) and discriminator(smallFOV/largeFOV)
generator = 'fcn32'
discriminator = 'largeFOV'
dataset = "std"

# Dataset specs
num_train_images = 565
num_val_images = 50
batch_size = 1
num_epoch_iterations = int(num_train_images/batch_size)
optimization_epochs = 20
num_iterations_to_alternate_trained_model = 500
num_classes = 8

# Options
save_model = 1
restore = 0
dataset_path = ""
restore_path = ""
save_path = ""

def adversarial_training(args, save_path = save_path):

    # Hyperparameters
    current_learning_rate_g, current_learning_rate_d, dropout, weight_decay,\
    loss_bce_weight, epochs, attempt = args
    initial_learning_rate_g = current_learning_rate_g
    initial_learning_rate_d = current_learning_rate_d
    epochs = int(epochs)

    def train_epoch(current_iteration):

        sess.run(reset_op)
        train_feed_dict = {updater_train.lr_g: current_learning_rate_g, updater_train.lr_d: current_learning_rate_d,
                           updater_train.pkeep: dropout, updater_train.weight_decay: weight_decay,
                           updater_train.loss_bce_weight: loss_bce_weight}
        batch_images = 1
        D_count = 0
        G_count = 0
        G_epoch_loss = 0
        D_epoch_loss = 0
        it = current_iteration

        while batch_images < num_epoch_iterations:
            # run x iterations of the discriminator
            while it < num_iterations_to_alternate_trained_model:
                _, D_tr_loss = sess.run([D_train_step, train_D_loss], feed_dict=train_feed_dict)
                for update_op in update_ops:
                    sess.run(update_op)
                it += 1
                D_count += 1
                batch_images += 1
                if np.isnan(D_tr_loss):
                    #print('got NaN as the loss value for 1 image in the discriminator')
                    D_epoch_loss += float('inf')
                else:
                    D_epoch_loss += D_tr_loss
                if batch_images == num_epoch_iterations:
                    normalised_G_epoch_loss = G_epoch_loss / G_count
                    normalised_D_epoch_loss = D_epoch_loss / D_count
                    return it, normalised_D_epoch_loss, normalised_G_epoch_loss, G_tr_pix_ac, G_train_mean_iou, G_train_class_acc

            # run x iterations of the generator
            while it < 2*num_iterations_to_alternate_trained_model:
                _, G_tr_loss, G_tr_pix_ac, G_train_mean_iou, G_train_class_acc = \
                    sess.run([G_train_step, train_G_loss, train_pixel_acc, train_mean_iou_acc, train_mean_per_class_acc],
                             feed_dict=train_feed_dict)
                it += 1
                G_count += 1
                batch_images += 1

                if np.isnan(G_tr_loss):
                    #print('got NaN as the loss value for 1 image in the generator')
                    G_epoch_loss += float('inf')
                else:
                    G_epoch_loss += G_tr_loss

                if batch_images == num_epoch_iterations:
                    normalised_G_epoch_loss = G_epoch_loss / G_count
                    normalised_D_epoch_loss = D_epoch_loss / D_count
                    return it, normalised_D_epoch_loss, normalised_G_epoch_loss, G_tr_pix_ac, G_train_mean_iou, G_train_class_acc
            it = 0

    def evaluate_model_on_val_set():
        sess.run(reset_op)
        val_feed_dict = {updater_val.pkeep: float(1.0), updater_val.weight_decay: weight_decay, updater_val.loss_bce_weight: loss_bce_weight}
        G_epoch_loss = 0
        D_epoch_loss = 0
        for i in range(num_val_images):
            D_te_loss, G_te_loss, G_val_pix_ac, G_val_mean_iou, G_val_class_acc = \
                sess.run([val_D_loss, val_G_loss, val_pixel_acc, val_mean_iou_acc, val_mean_per_class_acc],
                         feed_dict=val_feed_dict)
            if np.isnan(G_te_loss):
                #print('got NaN as the loss value for 1 image in the generator')
                G_epoch_loss += float('inf')
            else:
                G_epoch_loss += G_te_loss
            if np.isnan(D_te_loss):
                #print('got NaN as the loss value for 1 image in the discriminator')
                D_epoch_loss += float('inf')
            else:
                D_epoch_loss += D_te_loss
        normalised_G_epoch_loss = G_epoch_loss / num_val_images
        normalised_D_epoch_loss = D_epoch_loss / num_val_images
        return normalised_D_epoch_loss, normalised_G_epoch_loss, G_val_pix_ac, G_val_mean_iou, G_val_class_acc

    if generator == 'fcn32':
        train_images, train_labels = fcn_read_and_decode(dataset_path + 'train_dataset.tfrecords', epochs, batch_size)
        val_images, val_labels = fcn_read_and_decode_val_and_test(dataset_path + 'validation_dataset.tfrecords', epochs, 1)
    elif generator == 'unet':
        train_images, train_labels = unet_read_and_decode(dataset_path + 'train_dataset.tfrecords', epochs, batch_size)
        val_images, val_labels = unet_read_and_decode(dataset_path + 'validation_dataset.tfrecords', epochs, 1)
    elif generator == 'deeplab_v3':
        print('To be implemented...')
        sys.exit(1)

    # Define models
    with tf.variable_scope("model"):
        print('\nbuilding train model')
        updater_train = updater.GAN_trainer(x = train_images, y = train_labels, num_classes = num_classes, generator = generator, discriminator = discriminator, is_train = 1)
        G_train_step, D_train_step, train_G_loss, train_D_loss, train_pixel_acc, train_mean_iou_acc, train_mean_per_class_acc = updater_train.train_op()
    with tf.variable_scope("model", reuse=True):
        print('\nbuilding val model')
        updater_val = updater.GAN_trainer(x = val_images, y = val_labels, num_classes = num_classes, generator = generator, discriminator = discriminator, is_train = 0)
        _, _, val_G_loss, val_D_loss, val_pixel_acc, val_mean_iou_acc, val_mean_per_class_acc = updater_val.train_op()

    # Initialization
    config = tf.ConfigProto()
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess = tf.Session(config=config)
    sess.run(init_op)
    stream_vars = [i for i in tf.local_variables() if i.name.split('/')[1] == 'accuracy']
    reset_op = [tf.initialize_variables(stream_vars)]
    update_ops = tf.get_collection(SPECTRAL_NORM_UPDATE_OPS, scope='model/discriminator')

    # Saver
    saver = tf.train.Saver(max_to_keep=1)
    if restore is not 0:
        saver.restore(sess, restore_path)
        print('Loading checkpoint...')
    else:
        print('\nNo checkpoint file of basemodel found. Start from the scratch.\n')

    # Create a coordinator and run all QueueRunner objects
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    train_cost = []
    val_cost = []
    train_accuracy_pix = []
    train_accuracy_iou = []
    val_accuracy_pix = []
    val_accuracy_iou = []

    print("Starting to train...")
    sys.stdout.flush()
    train_iteration = 0
    try:
        epoch = 1
        best_val_iou = float(0.0)
        highest_val_iou_epoch = 0
        patience = 0
        while (True):

            # Check start time
            start = datetime.now()

            # Train for 1 epoch on training set
            train_iteration, D_tc, G_tc, ta_pix, ta_iou, ta_class = train_epoch(train_iteration)
            ta_class = np.mean(ta_class)
            train_cost.append([D_tc, G_tc])
            train_accuracy_pix.append(ta_pix)
            train_accuracy_iou.append(ta_iou)

            # Compute cost/accuracy on val dataset
            D_vc, G_vc, va_pix, va_iou, va_class = evaluate_model_on_val_set()
            va_class = np.mean(va_class)
            val_cost.append([D_vc, G_vc])
            val_accuracy_pix.append(va_pix)
            val_accuracy_iou.append(va_iou)

            # Print information
            print("\nEpoch: " + str(epoch))
            print('\ntrain cost dicriminator: ' + str(D_tc))
            print('train cost generator: ' + str(G_tc))
            print('train pixel accuracy: {:.2f}%'.format(ta_pix*100))
            print('train mean iou accuracy: {:.2f}%'.format(ta_iou*100))
            print('train mean per class accuracy: {:.2f}% \n'.format(ta_class*100))
            print('val cost discriminator: ' + str(D_vc))
            print('val cost generaror: ' + str(G_vc))
            print('val pixel accuracy: {:.2f}%'.format(va_pix*100))
            print('val mean iou accuracy: {:.2f}%'.format(va_iou*100))
            print('val mean per class accuracy: {:.2f}% \n'.format(va_class*100))

            if D_tc == float('inf') or G_tc == float('inf') or D_vc == float('inf') or G_vc == float('inf'):
                print('End of training due instability! \n')
                break

            # Check if loss is lower than in previous epochs
            if va_iou > best_val_iou:

                # Save model if specified
                if epochs > optimization_epochs and save_model == 1:
                    save_file = saver.save(sess, save_path + str(attempt) +"_adv_" + generator + "_model.ckpt", global_step=epoch)
                    print("Epoch %d - Model saved in path: %s \n" % (epoch, save_file))
                    sys.stdout.flush()

                best_val_iou = va_iou
                highest_val_iou_epoch = epoch
                patience = 0
            else:
                patience += 1
                # if 5 epochs without any improvement, decay learning rate
                if patience%5 == 0:
                    print('\nLearning rate decay\n')
                    current_learning_rate_g = lr_decay(current_learning_rate_g)
                    current_learning_rate_d = lr_decay(current_learning_rate_d)
                # if 30 epochs without any improvement, early stopping
                if patience == 30:
                    print('End of training due to early stopping! \n')
                    break

            # Increase epoch and decay learning rate
            epoch += 1

            # Check end time and compute epoch time lapse
            end = datetime.now()
            delta = end - start
            print('\tEpoch trained in %d hours, %d minutes and %d seconds' % (
                delta.seconds // 3600, ((delta.seconds // 60) % 60), (delta.seconds) % 60))
            sys.stdout.flush()

    except tf.errors.OutOfRangeError:
        print('End of training! \n')

    # Save accuracy and cost
    if epochs > optimization_epochs and save_model == 1:
        np.save('adversarial_'+generator+'_train_loss_'+dataset+'.npy', train_cost)
        np.save('adversarial_'+generator+'_val_loss_'+dataset+'.npy', val_cost)
        np.save('adversarial_'+generator+'_train_accuracy_pix_'+dataset+'.npy', train_accuracy_pix)
        np.save('adversarial_'+generator+'_val_accuracy_pix_'+dataset+'.npy', val_accuracy_pix)
        np.save('adversarial_'+generator+'_train_accuracy_iou_'+dataset+'.npy', train_accuracy_iou)
        np.save('adversarial_'+generator+'_val_accuracy_iou_'+dataset+'.npy', val_accuracy_iou)

    # Stop the threads
    coord.request_stop()

    # Wait for threads to stop
    coord.join(threads)
    sess.close()
    tf.reset_default_graph()

    print('Learning rate generator:', initial_learning_rate_g, 'Learning rate discriminator:', initial_learning_rate_d,
          ' Dropout:', dropout,
          'Weight Decay: ', weight_decay, ' Loss weight:', loss_bce_weight)
    print('\nBest model obtained in epoch ' + str(highest_val_iou_epoch) + '\n')
    print('----------------- \n')
    sys.stdout.flush()

    return np.max(val_accuracy_iou)

if __name__ == "__main__":

    # 1. Imports
    import tensorflow as tf
    import numpy as np

    search = "grid"  # "grid" or "random"

    if search == "random":

        # define a search space
        space = hp.choice('experiment number',
                             [
                                 (hp.uniform('learning_rate_generator', 0.00001, 0.001),
                                  hp.uniform('learning_rate_discriminator', 0.01, 0.5),
                                  hp.uniform('dropout_prob', 0.5, 1.0),
                                  hp.uniform('weight_decay', 1e-6, 1e-4),
                                  hp.uniform('loss_bce_weight', 0.01, 2.0),
                                  hp.quniform('Epochs',optimization_epochs, optimization_epochs+1, optimization_epochs))
                             ])


        best = fmin(adversarial_training, space, algo=tpe.suggest, max_evals=100)

        print('Best learning rate generator: ', best['learning_rate_generator'], 'Best learning rate discriminator: ',
               best['learning_rate_discriminator'], 'Best Dropout: ', best['dropout_prob'],
              'Best Weight Decay: ', best['weight_decay'], 'Best Loss Weight: ', best['loss_bce_weight'])
        print('-----------------\n')
        print('Starting training with optimized hyperparameters... \n')
        sys.stdout.flush()

        adversarial_training((best['learning_rate'], best['dropout_prob'],
                              best['weight_decay'], best['loss_bce_weight'],100))

    elif search == "grid":
        attempt = 1
        max_iou = float(0.0)
        learning_rate_generator = [1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
        learning_rate_discriminator = [0.01, 0.001, 0.0001, 0.1, 0.3 , 0.5]
        loss_bce_weight = [0.1, 0.5, 1]
        dropout_prob = [1.0]
        weight_decay = [0.0]
        for lr_g in learning_rate_generator:
            for lr_d in learning_rate_discriminator:
                for lbw in loss_bce_weight:
                    for wd in weight_decay:
                        iou = adversarial_training((lr_g, lr_d, 1.0, wd, lbw, 200, attempt))
                        attempt += 1
                        if (iou > max_iou):
                            best_lr_g = lr_g
                            best_lr_d = lr_d
                            best_lbw = lbw
                            best_wd = wd
                            max_iou = iou
        #adversarial_training((best_lr_g, best_lr_d, 1.0, best_wd, best_lbw, 300))
