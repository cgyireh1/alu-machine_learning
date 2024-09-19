#!/usr/bin/env python3
""" Layers """

import tensorflow as tf


def create_layer(prev, n, activation):
    """"
    Write the function def create_layer(prev, n, activation):
    prev is the tensor output of the previous layer
    n is the number of nodes in the layer to create
    activation is the activation function that the layer should use
    use tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG") 
    to implement He et. al initialization for the layer weights
    each layer should be given the name layer
    Returns: the tensor output of the layer
    """

    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n, activation=activation, kernel_initializer=initializer, name='layer')(prev)

    return layer