#!/usr/bin/env python3
""" Calculate accuracy of predictions """

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def calculate_accuracy(y, y_pred):
    """
    Calculates the accuracy of a prediction.

    Args:
        y: Placeholder for the true labels
        y_pred: Tensor containing the model's predictions

    Returns:
        Tensor containing the accuracy
    """
    correct_predictions = tf.equal(
        tf.argmax(y_pred, axis=1),
        tf.argmax(y, axis=1)
    )
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
