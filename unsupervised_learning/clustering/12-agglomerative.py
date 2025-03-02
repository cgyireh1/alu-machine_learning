#!/usr/bin/env python3
"""
A function that performs agglomerative
clustering on a dataset
"""


import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    X is a numpy.ndarray of shape (n, d) containing the dataset
    dist is the maximum cophenetic distance for all clusters
    Performs agglomerative clustering with Ward linkage
    Displays the dendrogram with each cluster displayed in a different color
    The only imports you are allowed to use are:
    import scipy.cluster.hierarchy
    import matplotlib.pyplot as plt
    Returns: clss, a numpy.ndarray of shape (n,) containing the cluster indices for each data point
    """
    return None
