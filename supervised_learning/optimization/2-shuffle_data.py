#!/usr/bin/env python3
""" Shuffle Data"""

import numpy as np


def shuffle_data(X, Y):
    """
    X is the first numpy.ndarray of shape (m, nx) to shuffle
    m is the number of data points
    nx is the number of features in X
    Y is the second numpy.ndarray of shape (m, ny) to shuffle
    m is the same number of data points as in X
    ny is the number of features in Y
    Returns: the shuffled X and Y matrices
    """
    perm = np.random.permutation(X.shape[0])
    X_shuffled = X[perm]
    Y_shuffled = Y[perm]

    return X_shuffled, Y_shuffled
