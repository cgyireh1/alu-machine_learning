#!/usr/bin/env python3
""" Early Stopping """


def early_stopping(cost, opt_cost, threshold, patience, count):
  """
  Early stopping should occur when the validation
  cost of the network has not decreased relative to the optimal
  validation cost by more than the threshold over a specific patience
  count cost is the current validation cost of the neural network
  opt_cost is the lowest recorded validation cost of the neural network
  threshold is the threshold used for early stopping
  patience is the patience count used for early stopping
  count is the count of how long the threshold has not been met
  """

  if opt_cost - cost > threshold:
      count = 0
  else:
      count += 1
  if count != patience:
      return False, count
  else:
      return True, count