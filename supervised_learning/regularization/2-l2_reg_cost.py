#!/usr/bin/env python3
"""L2 Regularization Cost"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    cost is a tensor containing the cost
    of the network without L2 regularization
    """

    L2_cost = cost +  tf.losses.get_regularization_losses()

    return L2_cost
