#!/usr/bin/env python3
"""
A function that finds the best number of clusters for a
GMM using the Bayesian Information Criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset
            n: the number of data points
            d: the number of dimensions for each data point
        kmin [positive int]:
            the minimum number of clusters to check for (inclusive)
        kmax [positive int]:
            the maximum number of clusters to check for (inclusive)
            if None, kmax should be set to maximum number of clusters possible
        iterations [positive int]:
            the maximum number of iterations for the algorithm
        tol [non-negative float]:
            the tolerance of the log likelihood, used for early stopping
        verbose [boolean]:
            determines if you should print information about the algorithm
    """
    return None, None, None, None
