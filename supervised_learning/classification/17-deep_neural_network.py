#!/usr/bin/env python3
""" DeepNeuralNetwork """

import numpy as np


class DeepNeuralNetwork():
    """
    A class that defines a deep neural network
    performing binary classification:
    """

    def __init__(self, nx, layers):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for k in range(self.L):
            if layers[k] < 1 or type(layers[k]) is not int:
                raise TypeError("layers must be a list of positive integers")
            self.weights["b" + str(k + 1)] = np.zeros((layers[k], 1))
            if k == 0:
                l1 = np.random.randn(layers[k], nx) * np.sqrt(2 / nx)
                self.weights["W" + str(k + 1)] = l1
            if k > 0:
                lay = np.sqrt(2 / layers[k - 1])
                l2 = np.random.randn(layers[k], layers[k - 1]) * lay
                He_lay = l2
                self.weights["W" + str(k + 1)] = He_lay

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
