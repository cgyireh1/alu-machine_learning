#!/usr/bin/env python3
""" Accuracy """


import tensorflow as tf

def calculate_accuracy(y, y_pred):
  """
  A function that calculates the accuracy of a prediction:
  y is a placeholder for the labels of the input data
  y_pred is a tensor containing the network’s predictions
  """

  prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
  accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
  return accuracy
