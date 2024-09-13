#!/usr/bin/env python3
"""Privatize Neuron"""

import numpy as np

def __init__(self, nx):
  """
  A class Neuron that defines a single neuron
  performing binary classification(Based on 0-neuron.py)
  """
  if type(nx) is not int:
     raise TypeError("nx must be an integer")
  if nx < 1:
     raise ValueError("nx must be a positive integer")
  self.__W = np.random.randn(1, nx)
  self.__b = 0
  self.__A = 0
