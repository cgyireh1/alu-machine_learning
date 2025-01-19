#!/usr/bin/env python3
"""
Bidirectional Cell Forward
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

        creates public instance attributes:
        Whf and bhf are for the hidden states in the forward direction
        Whb and bhb are for the hidden states in the backward direction
        Wy and by are for the outputs

        - weights should be initialized using random normal distribution
        - weights will be used on the right side for matrix multiplication
        - biases should be initialized as zeros
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
