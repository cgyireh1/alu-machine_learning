#!/usr/bin/env python3

"""
A function that calculates intra-cluster variance for a dataset
"""

import numpy as np


def variance(X, C):
    """
    X: numpy.ndarray (n, d) containing the dataset that
    will be used for K-means clustering
        - n no. of data points
        - d no. of dimensions for each data point
    C: numpy.ndarray (k, d) containing the centroid
        for each cluster
    return:
        - var: total intra-cluster variance
    """
    if not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray) or \
            len(X.shape) != 2 or len(C.shape) != 2 or \
            X.shape[1] != C.shape[1] or C.shape[1] <= 0 or X.size == 0 or \
            C.size == 0:
        return None

    dist_diff = np.linalg.norm(X - C[:, np.newaxis], axis=2).T
    minimum_dist = np.min(dist_diff, axis=1)
    var = np.sum(np.square(minimum_dist))
    return var
