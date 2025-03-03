#!/usr/bin/env python3

"""
A function that calculates the expectation
step in the EM algorithm for a GMM
"""

import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    X: numpy.ndarray (n, d) containing the dataset
        - n no. of data points
        - d no. of dimensions for each data point
    pi: numpy.ndarray (k,) containing the priors for each cluster
    m: numpy.ndarray (k, d) containing centroid means for each cluster
    S: numpy.ndarray (k, d, d) covariance matrices for each cluster
    return:
        - g: numpy.ndarray (k, n) containing the posterior
            probabilities for each data point in each cluster
        -l: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    try:
        g = np.zeros((k, n))
        for i in range(k):
            pdf_result = pdf(X, m[i], S[i])
            if pdf_result is None:
                return None, None
            g[i] = pi[i] * pdf_result

        total_likelihood = np.sum(g, axis=0)
        if np.any(total_likelihood == 0):
            return None, None

        likelihood = np.sum(np.log(total_likelihood))
        g /= total_likelihood
        return g, likelihood
    except Exception as e:
        return None, None
