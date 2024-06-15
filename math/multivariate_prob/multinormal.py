#!/usr/bin/env python3
"""
Multivariate Normal Distribution
"""


import numpy as np


class MultiNormal:
    """
    Class representing Multivariate Normal Distribution
    """
    def __init__(self, data):
        """
        data is a numpy.ndarray of shape (d, n)
        containing the data set:
        - d: number of dimensions in each data point
        - n: number of data points
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1).reshape(d, 1)
        X_mean = data - self.mean
        self.cov = np.matmul(X_mean, X_mean.T) / (n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point
        X: numpy.ndarray of shape (d, 1) containing the data point
        whose PDF should be calculated
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)"
                             .format(self.cov.shape[0]))
        if x.shape[0] != self.cov.shape[0] or x.shape[1] != 1:
            raise ValueError("x must have the shape ({}, 1)"
                             .format(self.cov.shape[0]))
        d = self.cov.shape[0]
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        X_mean = x - self.mean
        pdf = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        dist = np.matmul(np.matmul((X_mean).T, inv), (X_mean))
        pdf *= np.exp(-dist/2)
        pdf = pdf[0][0]
        return pdf
