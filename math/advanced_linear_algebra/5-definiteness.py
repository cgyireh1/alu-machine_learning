#!/usr/bin/env python3
"""
Function that calculates the definiteness of a matrix
"""


import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix
     matrix [numpy.ndarray of shape(n, n)]:else
     Return: the definiteness of the matrix or None:
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1] or \
       np.array_equal(matrix, matrix.T) is False:
        return None
    positive = 0
    negative = 0
    zero = 0
    eigenvalues = np.linalg.eig(matrix)[0]
    for value in eigenvalues:
        if value > 0:
            positive += 1
        if value < 0:
            negative += 1
        if value == 0 or value == 0.:
            zero += 1
    if positive and zero and negative == 0:
        return ("Positive semi-definite")
    elif negative and zero and positive == 0:
        return ("Negative semi-definite")
    elif positive and negative == 0:
        return ("Positive definite")
    elif negative and positive == 0:
        return ("Negative definite")
    return ("Indefinite")
