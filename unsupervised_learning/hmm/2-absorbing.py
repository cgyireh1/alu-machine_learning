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
    # check that P is the correct type and dimensions
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    # save value of n and check that P is square
    n, n_check = P.shape
    if n != n_check:
        return False
    return True
