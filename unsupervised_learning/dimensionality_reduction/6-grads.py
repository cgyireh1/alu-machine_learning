#!/usr/bin/env python3
"""
A function that calculates the gradients of Y
"""


import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
    parameters:
        Y [numpy.ndarray of shape (n, ndim)]:
            containing the low dimensional transformation of X (dataset)
            n: the number of data points
            ndim: the new dimensional representation of X
        P [numpy.ndarray of shape (n, n)]:
            containing the P affinities of X
    """
    return None
