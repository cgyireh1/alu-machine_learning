#!/usr/bin/env python3
"""Neuron Forward Propagation"""

import numpy as np


class Neuron():
    """
    A class Neuron that defines a single neuron performing
    binary classification (Based on 1-neuron.py):
    """

    def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        """The cost of the model using logistic regression"""
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A
    
    def cost(self, Y, A):
        """The cost of the model using logistic regression"""
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost / Y.shape[1]
