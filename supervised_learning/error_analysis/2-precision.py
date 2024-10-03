#!/usr/bin/env python3
""" Precision """

import numpy as np


def precision(confusion):
    """
    confusion is a confusion numpy.ndarray of shape
    (classes, classes) where row indices represent
    the correct labels and column indices
    represent the predicted labels
    classes is the number of classes
    Returns: a numpy.ndarray of shape (classes,)
    containing the precision of each class
    """

    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    Precision = TP / (TP + FP)
    return Precision
