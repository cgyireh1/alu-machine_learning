#!/usr/bin/env python3
""" Train """

import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op

def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    X_train is a numpy.ndarray containing the training input data
    Y_train is a numpy.ndarray containing the training labels
    X_valid is a numpy.ndarray containing the validation input data
    Y_valid is a numpy.ndarray containing the validation labels
    layer_sizes is a list containing the number of nodes in
    each layer of the network
    activations is a list containing the activation functions
    for each layer of the network
    alpha is the learning rate
    iterations is the number of iterations to train over
    save_path designates where to save the model
    Add the following to the graph’s collection
    placeholders x and y
    tensors y_pred, loss, and accuracy
    operation train_op
    After every 100 iterations, the 0th iteration, and iterations
    iterations, print the following:
    After {i} iterations: where i is the iteration
    \tTraining Cost: {cost} where {cost} is the training cost
    \tTraining Accuracy: {accuracy} where {accuracy} is the training accuracy
    \tValidation Cost: {cost} where {cost} is the validation cost
    \tValidation Accuracy: {accuracy} where {accuracy} is the validation accuracy
    Reminder: the 0th iteration represents the model before any training has occurred
    After training has completed, save the model to save_path
    You may use the following imports:
    calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
    calculate_loss = __import__('4-calculate_loss').calculate_loss
    create_placeholders = __import__('0-create_placeholders').create_placeholders
    create_train_op = __import__('5-create_train_op').create_train_op
    forward_prop = __import__('2-forward_prop').forward_prop
    """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    # Graph collection
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    y_pred = forward_prop(x, layer_sizes, activations)
    tf.add_to_collection('y_pred', y_pred)
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)
    # The train_op
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('train_op', train_op)
    
    saver = tf.train.Saver()
    # Start the session
    session = tf.Session()
    session.run(tf.global_variables_initializer())

    for i in range(iterations + 1):
        # Training the model
        train_cost = session.run(loss, feed_dict={x: X_train, y: Y_train})
        train_acc = session.run(accuracy, feed_dict={x: X_train, y: Y_train})
        validation_cost = session.run(loss, feed_dict={x: X_valid, y: Y_valid})
        validation_acc = session.run(accuracy, feed_dict={x: X_valid, y: Y_valid})
        # conditions
        if i % 100 == 0 or i == iterations:
            print("After {} iterations:".format(i))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(validation_cost))
            print("\tValidation Accuracy: {}".format(validation_acc))
        if i < iterations:
            session.run(train_op, feed_dict={x: X_train, y: Y_train})
    return saver.save(session, save_path)
