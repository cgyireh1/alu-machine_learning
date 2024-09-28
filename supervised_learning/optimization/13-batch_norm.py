#!/usr/bin/env python3
""" Batch Normalization """

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    Z is a numpy.ndarray of shape (m, n) that should be normalized
    m is the number of data points
    n is the number of features in Z
    gamma is a numpy.ndarray of shape (1, n)
    containing the scales used for batch normalization
    beta is a numpy.ndarray of shape (1, n)
    containing the offsets used for batch normalization
    epsilon is a small number used to avoid division by zero
    Returns: the normalized Z matrix
    """

    variance = Z.var(axis=0)
    mean = Z.mean(axis=0)
    normalized_Z = (Z - mean) / np.sqrt(variance + epsilon)
    Scaled_Z = gamma * normalized_Z + beta

    return Scaled_Z
