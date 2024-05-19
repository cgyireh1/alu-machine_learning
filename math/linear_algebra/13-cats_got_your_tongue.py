#!/usr/bin/env python3
""" function that concatenates two matrices along a specific axis """


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ returns new numpy.ndarray, the concatentation of two matrices """
    return np.concatenate((mat1, mat2), axis=axis)
