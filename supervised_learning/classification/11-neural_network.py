#!/usr/bin/env python3
"""Neural Network"""

import numpy as np


class NeuralNetwork():
    """ A class that defines a neural network
    with one hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__A1 = 1 / (1 + np.exp(-np.matmul(self.__W1, X) + self.__b1))
        self.__A2 = 1 / (1 + np.exp(-np.matmul(self.__W2, self.__A1)
                         + self.__b2))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        Cost = - (1 / Y.shape[1]) * np.sum(
            np.multiply(Y, np.log(A)) + np.multiply(
                1 - Y, np.log(1.0000001 - A)))
        return Cost
