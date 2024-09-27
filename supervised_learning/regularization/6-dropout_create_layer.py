#!/usr/bin/env python3
"""Create a Layer with Dropout"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    prev is a tensor containing the output of the previous layer
    n is the number of nodes the new layer should contain
    activation is activation function that should be used on the layer
    keep_prob is the probability that a node will be kept
    """

    dropout_layer = tf.keras.layers.Dropout(keep_prob)
    initiator_layer = tf.keras.contrib.layers.variance_scaling_initializer(mode="FAN_AVG"
                                                                          )

    dense_layer = tf.keras.layers.Dense(
      units=n, activation=activation,
      kernel_initializer=initiator_layer,
      kernel_regularizer=dropout_layer)(prev)
    return dense_layer
