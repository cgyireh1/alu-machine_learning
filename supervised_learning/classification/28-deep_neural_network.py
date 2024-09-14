#!/usr/bin/env python3
""" DeepNeuralNetwork """

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork():
    """
    A class that defines a deep neural network
    performing binary classification:
    """

    def __init__(self, nx, layers, activation='sig'):
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
        if activation != 'sig' and activation != 'tanh':
            raise ValueError("activation must be 'sig' or 'tanh'")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        self.__activation = activation
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

    @property
    def activation(self):
        return self.__activation

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        self.__cache["A0"] = X
        for lay in range(self.__L):
            weights = self.__weights
            cache = self.__cache
            activate = self.__activation
            Za = np.matmul(weights["W" + str(lay + 1)], cache["A" + str(lay)])
            Z = Za + weights["b" + str(lay + 1)]
            if lay == self.__L - 1:
                t = np.exp(Z)
                # Softmax activation
                cache["A" + str(lay + 1)] = (t / np.sum(
                    t, axis=0, keepdims=True))
            else:
                if activate == 'sig':
                    cache["A" + str(lay + 1)] = 1 / (1 + np.exp(-Z))
                else:
                    cache["A" + str(lay + 1)] = np.tanh(Z)
        return cache["A" + str(lay + 1)], cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """

        Cost = (-1 / Y.shape[1]) * np.sum(Y * np.log(A))
        return Cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        self.forward_prop(X)
        cache = self.__cache
        cost = self.cost(Y, cache["A" + str(self.__L)])
        mc = np.amax(cache["A" + str(self.__L)], axis=0)
        prediction = np.where(cache["A" + str(self.__L)] == mc, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        m = Y.shape[1]
        tW = self.__weights.copy()
        for i in reversed(range(self.__L)):
            A = self.__cache["A" + str(i + 1)]
            if i == self.__L - 1:
                dZ = self.__cache["A" + str(i + 1)] - Y
                dW = np.matmul(self.__cache["A" + str(i)], dZ.T) / m
            else:
                dW2 = np.matmul(tW["W" + str(i + 2)].T, dZ2)
                if self.__activation == 'sig':
                    gd = A * (1 - A)
                elif self.__activation == 'tanh':
                    gd = 1 - (A * A)
                dZ = dW2 * gd
                dW = np.matmul(dZ, self.__cache["A" + str(i)].T) / m
            # derivative of the loss with respect to b
            db3 = np.sum(dZ, axis=1, keepdims=True) / m
            if i == self.__L - 1:
                self.__weights["W" + str(i + 1)] = (tW["W" +
                                                       str(i + 1)] -
                                                    (alpha * dW).T)
            else:
                self.__weights["W" + str(i + 1)] = (tW["W" +
                                                       str(i + 1)] -
                                                    (alpha * dW))
            self.__weights["b" + str(i + 1)] = tW["b" + str(i + 1)] - (
                    alpha * db3)
            dZ2 = dZ

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the deep neural network """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")

        steps = 0
        # c_ax = np.zeros(iterations + 1)

        temp_cost = []
        temp_iterations = []
        for i in range(iterations + 1):
            self.forward_prop(X)
            cost = self.cost(Y, self.__cache["A" + str(self.__L)])
            if i % step == 0 or i == iterations:
                temp_cost.append(cost)
                temp_iterations.append(i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, cost))
            if i < iterations:
                self.gradient_descent(Y, self.__cache, alpha)

        if graph is True:
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.plot(temp_iterations, temp_cost)
            plt.show()
        return self.evaluate(X, Y)

    def save(self, filename):
        """
        The filename is the file to which the object should be saved
        If filename does not have the extension .pkl, add it
        """
        import pickle
        if '.pkl' not in filename:
            filename += '.pkl'

        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load(filename):
        """ Create the static method """
        import pickle
        try:
            with open(filename, 'rb') as f:
                Openfile = pickle.load(f)
            return Openfile
        except FileNotFoundError:
            return None
