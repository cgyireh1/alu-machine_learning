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

    for ti in time_step:
        H_f[ti+1] = bi_cell.forward(H_f[ti], X[ti])

    for ri in range(t-1, -1, -1):
        H_b[ri] = bi_cell.backward(H_b[ri+1], X[ri])
    H = np.concatenate((H_f[1:], H_b[:t]), axis=-1)

    Y = bi_cell.output(H)

    return H, Y
