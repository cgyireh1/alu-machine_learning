#!/usr/bin/env python3

"""
A function that initializes cluster
centroids for K-means
"""

import numpy as np


def initialize(X, k):
    """
    X: numpy.ndarray (n, d) containing the dataset that
    will be used for K-means clustering
        - n no. of data points
        - d no. of dimensions for each data point
    k: positive integer - the no. of clusters
    return: numpy.ndarray (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    n, d = X.shape
    min_val = X.min(axis=0)
    max_val = X.max(axis=0)

    return np.random.uniform(min_val, max_val, size=(k, d))
