#!/usr/bin/env python3
"""
One-Hot Encode
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    A function that converts a numeric label vector into a one-hot matrix
    """
    if type(Y) is not np.ndarray or len(Y.shape) != 1 or len(Y) == 1:
        return None
    if type(classes) is not int or classes != (Y.max() + 1):
        return None
    one_hot_enc = np.eye(classes)[Y].transpose()
    return one_hot_enc