#!/usr/bin/env python3
"""
LSTM Cell
"""


import numpy as np


class LSTMCell:
    """
    Represents a LSTM unit
    """
    def __init__(self, i, h, o):
        """
        parameters:
          i: dimensionality of the data
          h: dimensionality of the hidden state
          o: dimensionality of the outputs

        creates public instance attributes:
          Wf and bf are for the forget gate weights and biases
          Wu and bu are for the update gate weights and biases
          Wc and bc are for the intermediate cell state weights and biases
          Wo and bo are for the output gate weights and biases
          Wy and by are for the outputs weights and biases

        - weights should be initialized using random normal distribution
        - weights will be used on the right side for matrix multiplication
        - biases should be initialized as zeros
        """
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """
        Softmax function
          x: the value to perform softmax on to generate output of cell
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

    def forward(self, h_prev, c_prev, x_t):
        """
        Forward propagation for one time step

        parameters:
            h_prev [numpy.ndarray of shape (m, h)]:
                contains previous hidden state
                m: the batch size for the data
                h: dimensionality of hidden state
            c_prev [numpy.ndarray of shape (m, h)]:
                contains previous cell state
                m: the batch size for the data
                h: dimensionality of hidden state
            x_t [numpy.ndarray of shape (m, i)]:
                contains data input for the cell
                i: dimensionality of the data

        output of the cell should use softmax activation function

        returns:
            h_next, c_next, y:
            h_next: the next hidden state
            c_next: the next cell state
            y: the output of the cell
        """
        concat = np.concatenate((h_prev, x_t), axis=1)
        u_gate = self.sigmoid(np.matmul(concat, self.Wu) + self.bu)
        f_gate = self.sigmoid(np.matmul(concat, self.Wf) + self.bf)
        c_gate = np.tanh(np.matmul(concat, self.Wc) + self.bc)
        c_next = u_gate * c_gate + f_gate * c_prev
        o_gate = self.sigmoid(np.matmul(concat, self.Wo) + self.bo)
        h_next = o_gate * np.tanh(c_next)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, c_next, y
