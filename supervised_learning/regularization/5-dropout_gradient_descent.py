#!/usr/bin/env python3
"""Gradient Descent with Dropout"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Y is a one-hot numpy.ndarray of shape (classes, m)
    that contains the correct labels for the data
    classes is the number of classes
    m is the number of data points
    weights is a dictionary of the weights and biases of the network
    cache is a dictionary of the outputs and dropout masks of each
    layer of the neural network
    alpha is the learning rate
    keep_prob is the probability that a node will be kept
    L is the number of layers of the network
    All layers use the tanh activation function except the last,
    which uses the softmax activation function
    The weights of the network should be updated in place
    """

    W_copy = weights.copy()

    for i in reversed(range(L)):
        A = cache["A" + str(i + 1)]
        if i == L - 1:
            # d = derivative
            dW = (np.matmul(cache["A" + str(i)], (A - Y).T) / Y.shape[1]).T
            db = np.sum(dZ, axis=1, keepdims=True) / Y.shape[1]
        else:
            dW2 = np.matmul(W_copy["W" + str(i + 2)].T, dZ2)
            dZ = dW2 * (1 - (A * A)) * cache["D" + str(i + 1)] / keep_prob
            dW = np.matmul(dZ, cache["A" + str(i)].T) / Y.shape[1]
            db = np.sum(dZ, axis=1, keepdims=True) / Y.shape[1]
        weights["W" + str(i + 1)] = (W_copy["W" + str(i + 1)] - (alpha * dW))
        weights["b" + str(i + 1)] = W_copy["b" + str(i + 1)] - (alpha * db)
        dZ2 = dZ
