#!/usr/bin/env python3
""" Calculate the softmax cross-entropy loss """

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss.

    Args:
        y: Placeholder for the true labels
        y_pred: Tensor containing the network’s predictions

    Returns:
        Tensor containing the loss of the prediction
    """
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=y, logits=y_pred))
    return loss
