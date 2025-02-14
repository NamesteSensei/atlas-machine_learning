#!/usr/bin/env python3
"""Evaluate the output of a neural network."""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def evaluate(X, Y, save_path):
    """
    Evaluates the output of a neural network.

    Args:
        X: numpy.ndarray containing the input data to evaluate.
        Y: numpy.ndarray containing the one-hot labels for X.
        save_path: path to load the saved model from.

    Returns:
        The network's prediction, accuracy, and loss, respectively.
    """
    # Retrieve the tensors and operation from the collections
    x = tf.get_collection("x")[0]
    y = tf.get_collection("y")[0]
    y_pred = tf.get_collection("y_pred")[0]
    loss = tf.get_collection("loss")[0]
    accuracy = tf.get_collection("accuracy")[0]

    # Create a saver to restore the saved model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore the model from the given save_path
        saver.restore(sess, save_path)
        # Evaluate the predictions, accuracy, and loss using the restored graph
        pred, acc, cost = sess.run([y_pred, accuracy, loss],
                                   feed_dict={x: X, y: Y})
    return pred, acc, cost
