#!/usr/bin/env python3
""" Model """

import tensorflow as tf
import numpy as np


def shuffle_data(X, Y):
    """
    Function to shuffle data in a matrix
    """
    perm = np.random.permutation(X.shape[0])
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]

    return X_shuffled, Y_shuffled


def calculate_loss(y, y_pred):
    """
    Calculate the cross-entropy loss
    of a prediction
    """
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


def calculate_accuracy(y, y_pred):
    """
    Calculate the accuracy of a prediction in a DNN
    """
    prediction = tf.equal(tf.argmax(y, 1),
                                  tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return accuracy


def create_layer(prev, n, activation):
    """
    method to create a TF layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode='FAN_AVG')
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=initializer,
                            name='layer')(prev)
    return layer


def create_batch_norm_layer(prev, n, activation):
    """
    Function that normalized a batch in a DNN with Tf
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    x = tf.layers.Dense(units=n, activation=None,
                        kernel_initializer=initializer)
    x_prev = x(prev)
    scale = tf.Variable(tf.constant(1.0, shape=[n]), name='gamma')
    mean, variance = tf.nn.moments(x_prev, axes=[0])
    offset = tf.Variable(tf.constant(0.0, shape=[n]), name='beta')
    variance_epsilon = 1e-8

    normalization = tf.nn.batch_normalization(
        x_prev,
        mean,
        variance,
        offset,
        scale,
        variance_epsilon,
    )
    return activation(normalization)


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Forward propagation method using TF
    """
    layer = create_batch_norm_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        if i != len(layer_sizes) - 1:
            layer = create_batch_norm_layer(layer,
                                            layer_sizes[i],
                                            activations[i])
        else:
            layer = create_layer(layer, layer_sizes[i], activations[i])
    return layer


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(alpha,
                                       beta1,
                                       beta2,
                                       epsilon).minimize(loss)
    return optimizer


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Function that creates a learning rate decay operation
    """
    LR_op = tf.train.inverse_time_decay(alpha, global_step, decay_step,
                                        decay_rate, staircase=True)
    return LR_op


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Function that builds, trains, and saves a neural network
    """
    nx = Data_train[0].shape[1]
    classes = Data_train[1].shape[1]

    (X_train, Y_train) = Data_train
    (X_valid, Y_valid) = Data_valid

    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0)
    alpha_op = learning_rate_decay(alpha, decay_rate, global_step, 1)
    train_op = create_Adam_op(loss, alpha_op, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    initializer = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(initializer)
        if (X_train.shape[0] % batch_size) == 0:
            num_minibatches = int(X_train.shape[0] / batch_size)
            check = 1
        else:
            num_minibatches = int(X_train.shape[0] / batch_size) + 1
            check = 0

        for epoch in range(epochs + 1):
            feed_train = {x: X_train, y: Y_train}
            feed_valid = {x: X_valid, y: Y_valid}
            train_cost = sess.run(loss, feed_dict=feed_train)
            train_accuracy = sess.run(accuracy, feed_dict=feed_train)
            valid_cost = sess.run(loss, feed_dict=feed_valid)
            valid_accuracy = sess.run(accuracy, feed_dict=feed_valid)

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                Xs, Ys = shuffle_data(X_train, Y_train)

                for step_number in range(num_minibatches):
                    start = step_number * batch_size
                    end = (step_number + 1) * batch_size
                    if check == 0 and step_number == num_minibatches - 1:
                        x_minbatch = Xs[start::]
                        y_minbatch = Ys[start::]
                    else:
                        x_minbatch = Xs[start:end]
                        y_minbatch = Ys[start:end]

                    feed_mini = {x: x_minbatch, y: y_minbatch}
                    sess.run(train_op, feed_dict=feed_mini)

                    if ((step_number + 1) % 100 == 0) and (step_number != 0):
                        step_cost = sess.run(loss, feed_dict=feed_mini)
                        step_accuracy = sess.run(accuracy, feed_dict=feed_mini)
                        print("\tStep {}:".format(step_number + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))
            sess.run(tf.assign(global_step, global_step + 1))
            save_path = saver.save(sess, save_path)
    return save_path