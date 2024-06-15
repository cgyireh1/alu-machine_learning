#!/usr/bin/env python3
"""
Function that calculates mean and covariance of data sets
"""


import numpy as np


def mean_cov(X):
    """
    X: numpy.ndarray of shape (n, d) containing the data set:
      n: number of data points
      d: number of dimensions in each data point
    returns: mean, cov:
      mean is numpy.ndarray of shape (1, d)
      cov is numpy.ndarray of shape (d, d)
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')
    if X.shape[0] < 2:
        raise ValueError('X must contain multiple data points')
    n, d = X.shape
    mean = np.mean(X, axis=0).reshape(1, d)
    X_mean = X - mean
    cov = np.matmul(X_mean.T, X_mean) / (n - 1)
    return mean, cov