#!/usr/bin/env python3
""" Adam """

import numpy as np


def update_variables_Adam(alpha, beta1,
                          beta2, epsilon, var, grad, v, s, t):
    """
    alpha is the learning rate
    beta1 is the weight used for the first moment
    beta2 is the weight used for the second moment
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    v is the previous first moment of var
    s is the previous second moment of var
    t is the time step used for bias correction
    """

    V = beta1 * v + (1 - beta1) * grad
    V_corrected = V / (1 - beta1 ** t)
    S = beta2 * s + (1 - beta2) * np.sqrt(grad)
    S_corrected = s / (1 - beta2 ** t)

    updated_var = (var - alpha * V_corrected) / (np.sqrt(S_corrected
                                                       ) + epsilon)

    return updated_var, V, S
