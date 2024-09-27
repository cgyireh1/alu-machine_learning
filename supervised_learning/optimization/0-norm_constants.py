#!/usr/bin/env python3
"""Normalization Constants"""

import numpy as np # type: ignore


def normalization_constants(X):
    """
    X is the numpy.ndarray of shape (m, nx) to normalize
    m is the number of data points
    nx is the number of features
    """
  
    mean = np.mean(X, axis=0)
    stdev = np.std(X, axis=0)
    return mean, stdev
