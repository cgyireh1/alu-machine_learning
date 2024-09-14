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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        """
        self.__cache["A0"] = X
        for lay in range(self.__L):
            weights = self.__weights
            cache = self.__cache
            Za = np.matmul(weights["W" + str(lay + 1)], cache["A" + str(lay)])
            Z = Za + weights["b" + str(lay + 1)]
            cache["A" + str(lay + 1)] = 1 / (1 + np.exp(-Z))

        return cache["A" + str(self.__L)], cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        k = (1-Y)
        Cost = (-1 / m) * np.sum(Y * np.log(A) + k * (np.log(1.0000001 - A)))
        return Cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        cache = self.__cache
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network
        """
        weights = self.__weights.copy()
        A2 = self.__cache["A" + str(self.__L - 1)]
        A3 = self.__cache["A" + str(self.__L)]
        W3 = weights["W" + str(self.__L)]
        b3 = weights["b" + str(self.__L)]
        dZ_input = {}
        dZ_input["dz" + str(self.__L)] = A3 - Y

        # derivatives of the loss with respect to w and b
        dW3 = (1 / Y.shape[1]) * np.matmul(A2, (A3 - Y).T)
        db3 = (1 / Y.shape[1]) * np.sum((A3 - Y), axis=1, keepdims=True)

        self.__weights["W" + str(self.__L)] = W3 - (alpha * dW3).T
        self.__weights["b" + str(self.__L)] = b3 - (alpha * db3)

        for lay in range(self.__L - 1, 0, -1):
            cache = self.__cache
            Ap = cache["A" + str(lay - 1)]
            Wa = weights["W" + str(lay)]
            Wn = weights["W" + str(lay + 1)]
            ba = weights["b" + str(lay)]

            dZ2 = cache["A" + str(lay)] * (1 - cache["A" + str(lay)])
            dZ = np.matmul(Wn.T, dZ_input["dz" + str(lay + 1)]) * dZ2
            dW = (1 / Y.shape[1]) * np.matmul(Ap, dZ.T)
            db = (1 / Y.shape[1]) * np.sum(dZ, axis=1, keepdims=True)
            dZ_input["dz" + str(lay)] = dZ
            self.__weights["W" + str(lay)] = Wa - (alpha * dW).T
            self.__weights["b" + str(lay)] = ba - (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ Trains the deep neural network """

        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        steps = 0
        c_ax = np.zeros(iterations + 1)

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

        fileObject = open(filename, 'wb')
        pickle.dump(self, fileObject)
        fileObject.close()

    @staticmethod
    def load(filename):
        """ Create the static method """
        try:
            with open(filename, 'rb') as f:
                fileOpen = pkl.load(f)
            return fileOpen
        except FileNotFoundError:
            return None
