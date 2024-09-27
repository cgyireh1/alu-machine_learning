#!/usr/bin/env python3
"""Forward Propagation with Dropout"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    X is a numpy.ndarray of shape (nx, m)
    containing the input data for the network
    nx is the number of input features
    m is the number of data points
    weights is a dictionary of the weights and biases of the network
    L the number of layers in the network
    keep_prob is the probability that a node will be kept
    All layers except the last should use tanh activation function
    The last layer should use the softmax activation function
    """

    cache = {}
    cache['A0'] = X
    for i in range(L):
        W = weights["W" + str(i + 1)]
        A = cache["A" + str(i)]
        B = weights["b" + str(i + 1)]
        Z = np.matmul(W, A) + B
        dropout = np.random.rand(Z.shape[0], Z.shape[1])
        dropout = np.where(dropout < keep_prob, 1, 0)

        if i == L - 1:
            softmax = np.exp(Z)
            cache["A" + str(i + 1)] = (softmax / np.sum(softmax, axis=0,
                                                        keepdims=True))
        else:
            tanh = np.tanh(Z)
            cache["A" + str(i + 1)] = tanh
            cache["D" + str(i + 1)] = dropout
            cache["A" + str(i + 1)] *= dropout
            cache["A" + str(i + 1)] /= keep_prob
    return cache
