#!/usr/bin/env python3
"""
One-Hot Encode
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    A function that converts a numeric label vector into a one-hot matrix
    """
    if len(Y) == 0:
        return None
    elif type(classes) is not int:
        return None
    elif not isinstance(Y, np.ndarray):
        return None
    elif classes <= np.amax(Y):
        return None

    one_hot_enc = np.zeros((classes, len(Y)))
    one_hot_enc[Y, np.arange(len(Y))] = 1
    return one_hot_enc
