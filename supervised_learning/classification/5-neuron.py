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

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        cost = self.cost(Y, self.forward_prop(X))
        prediction = np.where(self.forward_prop(X) >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        # derivatives of loss with respect to W (dW) and b (db)
        dW = np.matmul(A - Y, X.T) / X.shape[1]
        db = np.sum(A - Y) / X.shape[1]
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
        return self.__W, self.__b
