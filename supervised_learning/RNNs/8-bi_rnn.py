#!/usr/bin/env python3
"""
Bidirectional RNN
"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN
    """
    t, m, i = X.shape
    time_step = range(t)

    _, h = h_0.shape

    H_f = np.zeros((t+1, m, h))
    H_b = np.zeros((t+1, m, h))

    H_f[0] = h_0
    H_b[t] = h_t

    for i in time_step:
        H_f[i+1] = bi_cell.forward(H_f[i], X[i])
        H_b[t-i] = bi_cell.backward(H_b[t-i+1], X[t-i])

    H = np.concatenate((H_f, H_b), axis=0)

    Y = bi_cell.output(H)

    return H, Y
