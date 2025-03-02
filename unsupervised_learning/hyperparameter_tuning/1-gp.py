#!/usr/bin/env python3
"""
Gaussian Process Prediction
"""

import numpy as np


class GaussianProcess():
    """
    Class constructor that represents a noiseless 1D Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        The Init method
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices
        """
        sqdist1 = np.sum(X1 ** 2, 1).reshape(-1, 1) + np.sum(X2 ** 2, 1)
        sqdist2 = 2 * np.dot(X1, X2.T)
        sqdist = sqdist1 - sqdist2
        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """
        Method that predicts the mean and
        standard deviation of points in a gaussian process
        """
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        # formula mu: μ∗ =K∗.T Ky^−1y
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu_s = np.reshape(mu_s, -1)
        # formula sigma: Σ∗ =K∗∗ − K∗.T Ky^−1 K∗
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        cov_s = cov_s.diagonal()

        return mu_s, cov_s
