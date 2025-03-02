#!/usr/bin/env python3
"""
A function that calculates the cost of the t-SNE transformation
"""


import numpy as np


def cost(P, Q):
    """
    parameters:
        P [numpy.ndarray of shape (n, n)]:
            containing the P affinities
        Q [numpy.ndarray of shape (n, n)]:
            containing the Q affinities
    returns:
        c: the cost of the transformation
    """
    Q = np.where(Q == 0, 1e-12, Q)
    P = np.where(P == 0, 1e-12, P)
    c = np.sum(P * np.log(P / Q))
    return c
