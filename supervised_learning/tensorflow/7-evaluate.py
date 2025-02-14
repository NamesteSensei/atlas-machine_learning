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
        A tuple: (network's prediction, accuracy, loss).
    """
    # Retrieve tensors from the graph collections
    x = tf.get_collection("x")[0]
    y = tf.get_collection("y")[0]
    y_pred = tf.get_collection("y_pred")[0]
    loss = tf.get_collection("loss")[0]
    accuracy = tf.get_collection("accuracy")[0]

    # Create a Saver to restore the saved model
    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore the model from the provided checkpoint
        saver.restore(sess, save_path)
        # Evaluate prediction, accuracy, and loss
        pred, acc, cost = sess.run([y_pred, accuracy, loss],
                                   feed_dict={x: X, y: Y})
    return pred, acc, cost
