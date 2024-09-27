#!/usr/bin/env python3
"""Gradient Descent with L2 Regularization"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y is a one-hot numpy.ndarray of shape (classes, m) 
    that contains the correct labels for the data
    classes is the number of classes
    m is the number of data points
    weights is a dictionary of the weights and biases of the network
    cache is a dictionary of the outputs of each layer of the network
    alpha is the learning rate
    lambtha is the L2 regularization parameter
    L is the number of layers of the network
    """

    W_copy = weights.copy()

    for i in reversed(range(L)):
        A = cache["A" + str(i + 1)]
        if i == L - 1:
            dZ = cache["A" + str(i + 1)] - Y
            dW = (np.matmul(cache["A" + str(i)], dZ.T) / Y.shape[1]).T
            dW_L2 = dW + (lambtha / Y.shape[1]) * W_copy["W" + str(i + 1)]
            db = np.sum(dZ, axis=1, keepdims=True) / Y.shape[1]
        else:
            dW2 = np.matmul(W_copy["W" + str(i + 2)].T, dZ2)
            tanh = 1 - (A * A)
            dZ = dW2 * tanh
            dW = np.matmul(dZ, cache["A" + str(i)].T) / Y.shape[1]
            dW_L2 = dW + (lambtha / Y.shape[1]) * W_copy["W" + str(i + 1)]
            db = np.sum(dZ, axis=1, keepdims=True) / Y.shape[1]
        weights["W" + str(i + 1)] = (W_copy["W" + str(i+1)] - (alpha * dW_L2))
        weights["b" + str(i + 1)] = W_copy["b" + str(i + 1)] - (alpha * db)
        dZ2 = dZ
