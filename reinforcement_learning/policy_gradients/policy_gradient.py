#!/usr/bin/env python3
"""
Monte Carlo policy gradient
"""


import numpy as np


def policy(matrix, weight):
    """
    Computes policy with a weight of a matrix
    """
    dot_product = matrix.dot(weight)
    exp = np.exp(dot_product)
    policy = exp / np.sum(exp)
    return policy


def policy_gradient(state, weight):
    """
    Computes the Monte Carlo policy gradient based on the policy
        calculated from the above policy() function
    """

    Policy = policy(state, weight)
    action = np.random.choice(len(Policy[0]), p=Policy[0])
    s = Policy.reshape(-1, 1)
    softmax = (np.diagflat(s) - np.dot(s, s.T))[action, :]
    dlog = softmax / Policy[0, action]

    gradient = state.T.dot(dlog[None, :])
    return action, gradient
