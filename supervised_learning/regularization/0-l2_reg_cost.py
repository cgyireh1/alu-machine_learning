#!/usr/bin/env python3
"""L2 Regularization Cost"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    cost is the cost of the network without L2 reg
    lambtha is the regularization parameter
    weights is a dictionary of the weights
    and biases of the neural network
    L is the number of layers in the neural network
    m is the number of data points used
    """

    l2_reg = 0
    for key, values in weights.items():
        if key[0] == 'W':
            l2_reg += np.linalg.norm(values)
    L2_cost = cost + (lambtha / (2 * m) * l2_reg)
    return L2_cost
