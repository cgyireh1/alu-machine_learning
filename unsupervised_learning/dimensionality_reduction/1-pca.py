#!/usr/bin/env python3
'''
Write a function that performs PCA on a dataset
'''


import numpy as np


def pca(X, ndim):
    '''
    PCA v2
    Performs PCA on a dataset
    '''
    mean = np.mean(X, axis=0, keepdims=True)
    A = X - mean
    u, s, v = np.linalg.svd(A)
    W = v.T[:, :ndim]
    T = np.matmul(A, W)
    return (T)
