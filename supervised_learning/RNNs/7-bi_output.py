#!/usr/bin/env python3
"""
Defines the class BidirectionalCell that represents a bidirectional RNN cell
"""


import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional RNN cell
    """
    def __init__(self, i, h, o):
        """
        parameters:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs
        """
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Whf = np.random.normal(size=(h + i, h))
        self.Whb = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=((2 * h), o))

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step
        parameters:
            h_prev [numpy.ndarray of shape (m, h)]:
                contains previous hidden state
                m: the batch size for the data
                h: dimensionality of hidden state
            x_t [numpy.ndarray of shape (m, i)]:
                contains data input for the cell
                m: the batch size for the data
                i: dimensionality of the data

        returns:
            h_next: the next hidden state
        """
        h_x = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_x, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step
        parameters:
            h_next [numpy.ndarray of shape (m, h)]:
                contains the next hidden state
                m: the batch size for the data
                h: dimensionality of hidden state
            x_t [numpy.ndarray of shape (m, i)]:
                contains data input for the cell
                i: dimensionality of the data
        returns:
            h_prev: the previous hidden state
        """
        h_x = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(h_x, self.Whb) + self.bhb)

        return h_prev

    def softmax(self, x):
        """
        Softmax function
          x: the value to perform softmax on to generate output of cell
        """
        a_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = a_x / a_x.sum(axis=1, keepdims=True)
        return softmax

    def output(self, H):
        """
        Calculates all outputs for the RNN

        parameters:
            H [numpy.ndarray of shape (t, m, 2 * h)]:
                contains the concatenated hidden states from both directions,
                    excluding their initialized states
                t: number of time steps
                m: the batch size for the data
                h: the dimensionality of the hidden states

        returns:
            Y: the outputs
        """
        t, m, h = H.shape

        Y = []

        for step in range(t):
            y = self.softmax(np.matmul(H[step], self.Wy) + self.by)
            Y.append(y)

        return np.array(Y)
