#!/usr/bin/env python3
""" Model """

import tensorflow as tf


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Data_train is a tuple containing the training inputs and training labels, respectively
    Data_valid is a tuple containing the validation inputs and validation labels, respectively
    layers is a list containing the number of nodes in each layer of the network
    activation is a list containing the activation functions used for each layer of the network
    alpha is the learning rate
    beta1 is the weight for the first moment of Adam Optimization
    beta2 is the weight for the second moment of Adam Optimization
    epsilon is a small number used to avoid division by zero
    decay_rate is the decay rate for inverse time decay of the learning rate (the corresponding decay step should be 1)
    batch_size is the number of data points that should be in a mini-batch
    epochs is the number of times the training should pass through the whole dataset
    save_path is the path where the model should be saved to
    Returns: the path where the model was saved
    Your training function should allow for a smaller final batch (a.k.a. use the entire training set)
    the learning rate should remain the same within the an epoch (a.k.a. all mini-batches within an epoch should use the same learning rate)
    Before each epoch, you should shuffle your training data
    Before the first epoch and after every subsequent epoch, the following should be printed:
    After {epoch} epochs: where {epoch} is the current epoch
    \tTraining Cost: {train_cost} where {train_cost} is the cost of the model on the entire training set
    \tTraining Accuracy: {train_accuracy} where {train_accuracy} is the accuracy of the model on the entire training set
    \tValidation Cost: {valid_cost} where {valid_cost} is the cost of the model on the entire validation set
    \tValidation Accuracy: {valid_accuracy} where {valid_accuracy} is the accuracy of the model on the entire validation set
    After every 100 steps of gradient descent within an epoch, the following should be printed:
    \tStep {step_number}: where {step_number} is the number of times gradient descent has been run in the current epoch
    \t\tCost: {step_cost} where {step_cost} is the cost of the model on the current mini-batch
    \t\tAccuracy: {step_accuracy} where {step_accuracy} is the accuracy of the model on the current mini-batch
    """