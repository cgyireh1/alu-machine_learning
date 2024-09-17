#!/usr/bin/env python3
""" One-Hot Decode """


import numpy as np


def one_hot_decode(one_hot):
    """
    A function that converts a one-hot matrix into a vector of labels
    """
    if not isinstance(one_hot, np.ndarray):
        return None
    elif len(one_hot) == 0:
        return None
    elif len(one_hot.shape) != 2:
        return None
    elif not np.all((one_hot == 0) | (one_hot == 1)):
        return None
    return np.argmax(one_hot, axis=0)
