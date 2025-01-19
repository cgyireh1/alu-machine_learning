#!/usr/bin/env python3
"""
RNN Cell
"""


import numpy as np


class RNNCell:
    """
    Create the class RNNCell that represents a cell of a simple RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor
          i: dimensionality of the data
          h: dimensionality of the hidden state
          o: dimensionality of the outputs

        creates public instance attributes:
          Wh and bh are for the concatenated hidden state and input data
          Wy and by: output weights and biases

        - weights should be initialized using random normal distribution
        - weights will be used on the right side for matrix multiplication
        - biases should be initiliazed as zeros
        """
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """
        Softmax function
          x: the value to perform softmax on to generate output of cell
        """
        a_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = a_x / a_x.sum(axis=1, keepdims=True)
        return softmax

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step
          h_prev [numpy.ndarray of shape (m, h)]:
              contains previous hidden state
              m: the batch size for the data
              h: dimensionality of hidden state
          x_t [numpy.ndarray of shape (m, i)]:
              contains data input for the cell
              m: the batch size for the data
              i: dimensionality of the data
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
