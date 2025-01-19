#!/usr/bin/env python3
"""
GRU Cell
"""


import numpy as np


class GRUCell:
    """
    Represents a gated recurrent unit
    """
    def __init__(self, i, h, o):
        """
        Class constructor
          i: dimensionality of the data
          h: dimensionality of the hidden state
          o: dimensionality of the outputs

        creates public instance attributes:
          Wz and bz: update gate weights and biases
          Wr and br: reset gate weights and biases
          Wh and bh: intermediate hidden state and input data weights and bias
          Wy and by are for the output weights and biases

        - weights should be initialized using random normal distribution
        - weights will be used on the right side for matrix multiplication
         biases should be initialized as zeros
        """
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.Wz = np.random.normal(size=(h + i, h))
        self.Wr = np.random.normal(size=(h + i, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """
        Softmax function
          x: value to perform softmax on to generate output of cell
        """
        a_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = a_x / a_x.sum(axis=1, keepdims=True)
        return softmax

    def sigmoid(self, x):
        """
        Sigmoid function
          x: the value to perform sigmoid on
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

    def forward(self, h_prev, x_t):
        """
        Forward propagation for one time step

          h_prev [numpy.ndarray of shape (m, h)]:
              contains previous hidden state
              m: the batch size for the data
              h: dimensionality of hidden state
          x_t [numpy.ndarray of shape (m, i)]:
              contains data input for the cell
              i: dimensionality of the data

        output of the cell should use softmax activation function

        returns: h_next, y
            h_next: next hidden state
            y: output of the cell
        """
        concat1 = np.concatenate((h_prev, x_t), axis=1)
        z_gate = self.sigmoid(np.matmul(concat1, self.Wz) + self.bz)
        r_gate = self.sigmoid(np.matmul(concat1, self.Wr) + self.br)

        concat2 = np.concatenate((r_gate * h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat2, self.Wh) + self.bh)
        h_next *= z_gate
        h_next += (1 - z_gate) * h_prev

        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
