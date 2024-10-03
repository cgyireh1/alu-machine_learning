#!/usr/bin/env python3
""" Specificity """

import numpy as np


def specificity(confusion):
  """
  confusion is a confusion numpy.ndarray of shape
  (classes, classes) where row indices represent
  the correct labels and column indices
  represent the predicted labels
  classes is the number of classes
  Returns: a numpy.ndarray of shape (classes,)
  containing the specificity of each class
  """
  TP = np.diag(confusion)
  FP = np.sum(confusion, axis=0) - TP
  FN = np.sum(confusion, axis=1) - TP
  TN = np.sum(confusion) - (FP + FN + TP)

  Specificity = TN / (TN + FP)
  return Specificity