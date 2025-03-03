#!/usr/bin/env python3

"""
A function that initializes variables for
a Gaussian Mixture Model
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    X: numpy.ndarray (n, d) containing the dataset
        - n no. of data points
        - d no. of dimensions for each data point
    k: positive integer - the number of clusters
    return:
        - pi: numpy.ndarray (k,) containing priors for each cluster
        initialized to be equal
        - m: numpy.ndarray (k, d) containing centroid means for each cluster,
        initialized with K-means
        - S: numpy.ndarray (k, d, d) covariance matrices for each cluster,
        initialized as identity matrices
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None
    C, clss = kmeans(X, k)
    pi = np.full(k, 1 / k)
    m = C
    S = np.tile(np.identity(X.shape[1]), (k, 1, 1))
    return pi, m, S
