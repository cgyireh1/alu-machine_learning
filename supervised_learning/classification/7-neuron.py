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

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """Trains the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        steps = 0

        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A)
            if (i == steps or i == iterations) and step:
                print("Cost after {} iterations: {}".format(i, cost))
                steps += step
            if i < iterations:
                self.gradient_descent(X, Y, self.__A, alpha)
            if graph is True:
                np.zeros(iterations + 1)[i] = cost
        if graph is True:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(np.arange(0, iterations + 1), np.zeros(iterations + 1))
            plt.show()
        return self.evaluate(X, Y)
