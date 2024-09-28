#!/usr/bin/env python3
""" Batch Normalization Upgraded """

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should be used on the output of the layer
    you should use the tf.layers.Dense layer as the base layer with kernal initializer tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    your layer should incorporate two trainable parameters, gamma and beta, initialized as vectors of 1 and 0 respectively
    you should use an epsilon of 1e-8
    """

    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    x = tf.layers.Dense(units=n, activation=None, kernel_initializer=initializer)
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