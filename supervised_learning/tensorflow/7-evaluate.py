#!/usr/bin/env python3
""" Evaluate """

import tensorflow as tf


def evaluate(X, Y, save_path):
    """
    X is a numpy.ndarray containing the input data to evaluate
    Y is a numpy.ndarray containing the one-hot labels for X
    save_path is the location to load the model from
    Import the meta graph
    Get the following from the graph’s collection:
    tensors y_pred, loss, and accuracy
    Returns: the network’s prediction, accuracy, and loss
    """

    with tf.Session() as session:
        saver = tf.train.import_meta_graph("{}.meta".format(save_path))
        saver.restore(session, save_path)

        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        y_pred = tf.get_collection("y_pred")[0]
        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]

        feed_dict = {x: X, y: Y}
        evaluate_y_pred = session.run(y_pred, feed_dict=feed_dict)
        evaluate_accuracy = session.run(accuracy, feed_dict=feed_dict)
        evaluate_loss = session.run(loss, feed_dict=feed_dict)

        return evaluate_y_pred, evaluate_accuracy, evaluate_loss
