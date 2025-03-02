#!/usr/bin/env python3
'''
A function that determines if a markov chain
is absorbing
'''


import numpy as np


def absorbing(P):
    '''
    Args:
        P: 2D numpy.ndarray of shape (n, n) representing the transition matrix
           P[i, j]: is the probability of transitioning from state i to state j
           n: the number of states in the markov chain
    Returns: True if it is absorbing, or False on failure
    '''
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    n, n = P.shape
    if n != P.shape[0]:
        return False
    if np.sum(P, axis=1).all() != 1:
        return False
    if np.any(np.diag(P) == 1):
        return True
    return False
