#!/usr/bin/env python3
"""Neural Network"""

import numpy as np


class NeuralNetwork():
    """ A class that defines a neural network
    with one hidden layer performing binary classification """

    def __init__(self, nx, nodes):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

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
        """ Calculates the forward propagation of the neural network """
        self.__A1 = 1 / (1 + np.exp(-(np.matmul(self.__W1, X) + self.__b1)))
        self.__A2 = 1 / (1 + np.exp(-(np.matmul(self.__W2, self.__A1)
                         + self.__b2)))

        return self.__A1, self.__A2

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        Cost = - (1 / Y.shape[1]) * np.sum(
            np.multiply(Y, np.log(A)) + np.multiply(
                1 - Y, np.log(1.0000001 - A)))
        return Cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions
        """
        self.forward_prop(X)
        cost = self.cost(Y, self.__A2)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        # Derivative of z2 and z1 with respect to A2
        dz1 = np.matmul(self.__W2.T, (A2 - Y)) * (A1 * (1 - A1))
        dz2 = A2 - Y

        self.__W1 -= alpha * np.matmul(dz1, X.T) / Y.shape[1]
        self.__b1 -= alpha * np.sum(dz1, axis=1, keepdims=True) / Y.shape[1]
        self.__W2 -= (alpha * np.matmul(A1, dz2.T) / Y.shape[1]).T
        self.__b2 -= alpha * np.sum(dz2, axis=1, keepdims=True) / Y.shape[1]

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the neural network
        """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        steps = 0

        temp_cost = []
        temp_iterations = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__A2)
            if i % step == 0 or i == iterations:
                temp_cost.append(cost)
                temp_iterations.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)


        if graph is True:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(temp_iterations, temp_cost)
            plt.show()
        return self.evaluate(X, Y)
