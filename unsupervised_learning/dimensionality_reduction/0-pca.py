#!/usr/bin/env python3
'''
Writes a function that performs PCA on a dataset
'''


import numpy as np


def pca(X, var=0.95):
    '''
    performs PCA on a dataset
    '''
    u, s, v = np.linalg.svd(X)
    ratios = list(x / np.sum(s) for x in s)
    variance = np.cumsum(ratios)
    nd = np.argwhere(variance >= var)[0, 0]
    W = v.T[:, :(nd + 1)]
    return (W)
