#!/usr/bin/env python3
"""Create a Layer with L2 Regularization"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
  """
  prev is a tensor containing output of the previous layer
  n is the number of nodes the new layer should contain
  activation is the activation function that
  should be used on the layer
  lambtha is the L2 regularization parameter
  """

  regularizer = tf.contrib.layers.l2_regularizer(lambtha)
  initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
  tensor = tf.layers.Dense(units=n, activation=activation,
                           kernel_initializer=initializer,
                           kernel_regularizer=regularizer)
  return tensor(prev)