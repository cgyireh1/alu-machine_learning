#!/usr/bin/env python3
""" Mini-Batch """

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid,
                     Y_valid, batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    X_train is a numpy.ndarray of shape (m, 784)
    containing the training data
    m is the number of data points
    784 is the number of input features
    Y_train is a one-hot numpy.ndarray of shape (m, 10)
    containing the training labels
    10 is the number of classes the model should classify
    X_valid is a numpy.ndarray of shape (m, 784)
    containing the validation data
    Y_valid is a one-hot numpy.ndarray of shape (m, 10)
    containing the validation labels
    batch_size is the number of data points in a batch
    epochs is the number of times the training should
    pass through the whole dataset
    load_path is the path from which to load the model
    save_path is the path to where the model should be
    saved after training
    Returns: the path where the model was saved
    Your training function should allow for a
    smaller final batch (a.k.a. use the entire training set)
    1) meta graph and restore session
    2) Get the following tensors and ops from the collection restored
    x is a placeholder for the input data
    y is a placeholder for the labels
    accuracy is an op to calculate the accuracy of the model
    loss is an op to calculate the cost of the model
    train_op is an op to perform one pass of gradient
    descent on the model
    3) loop over epochs:
    shuffle data
    loop over the batches:
    get X_batch and Y_batch from data
    train your model
    """

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(sess, load_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        if (X_train.shape[0] % batch_size) == 0:
            minibatches = int(X_train.shape[0] / batch_size)
            check = 1
        else:
            minibatches = int(X_train.shape[0] / batch_size) + 1
            check = 0

        for epoch in range(epochs + 1):
            feed_train = {x: X_train, y: Y_train}
            feed_valid = {x: X_valid, y: Y_valid}
            train_cost = sess.run(loss, feed_dict=feed_train)
            train_accuracy = sess.run(accuracy, feed_dict=feed_train)
            valid_cost = sess.run(loss, feed_dict=feed_valid)
            valid_accuracy = sess.run(accuracy, feed_dict=feed_valid)

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                Xs, Ys = shuffle_data(X_train, Y_train)

                for step_number in range(minibatches):
                    start = step_number * batch_size
                    end = (step_number + 1) * batch_size
                    if check == 0 and step_number == minibatches - 1:
                        x_minbatch = Xs[start::]
                        y_minbatch = Ys[start::]
                    else:
                        x_minbatch = Xs[start:end]
                        y_minbatch = Ys[start:end]

                    feed_mini = {x: x_minbatch, y: y_minbatch}
                    sess.run(train_op, feed_dict=feed_mini)

                    if ((step_number + 1) % 100 == 0) and (step_number != 0):
                        step_cost = sess.run(loss, feed_dict=feed_mini)
                        step_accuracy = sess.run(accuracy, feed_dict=feed_mini)
                        print("\tStep {}:".format(step_number + 1))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))
        save_path = saver.save(sess, save_path)
    return save_path
