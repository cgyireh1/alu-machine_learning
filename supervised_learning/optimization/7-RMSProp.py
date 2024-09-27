#!/usr/bin/env python3
""" RMSProp """

import numpy as np # type: ignore


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Alpha is the learning rate
    beta2 is the RMSProp weight
    epsilon is a small number to avoid division by zero
    var is a numpy.ndarray containing the variable to be updated
    grad is a numpy.ndarray containing the gradient of var
    s is the previous second moment of var
    Returns: the updated variable and the new moment, respectively
    """
  
    Sdv = (beta2 * s) + ((1 - beta2) * np.square(grad))
    new_Var = var - alpha * (grad / np.sqrt(Sdv) + epsilon)
    return new_Var, Sdv
