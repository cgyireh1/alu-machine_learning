#!/usr/bin/env python3
"""Function that calculates a correlation matrix"""


import numpy as np


def correlation(C):
    """
    C: numpy.ndarray of shape (d, d)
    containing a covariance matrix

    Returns: numpy.ndarray of shape (d, d)
    containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2:
        raise ValueError('C must be a 2D square matrix')
    if C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    Std_D = np.sqrt(np.diag(C))
    Std_D_inverse = 1 / np.outer(Std_D, Std_D)
    corr = Std_D_inverse * C
    return corr