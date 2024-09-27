#!/usr/bin/env python3
""" Momentum Upgraded """

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
  """
  loss is the loss of the network
  alpha is the learning rate
  beta1 is the momentum weight
  Returns: the momentum optimization operation
  """

  optimizer = tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
  return optimizer