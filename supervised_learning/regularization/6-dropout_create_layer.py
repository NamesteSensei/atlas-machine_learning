#!/usr/bin/env python3
"""
Create a layer with Dropout
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using dropout.

    Parameters
    ----------
    prev : tf.Tensor
        Output of the previous layer.
    n : int
        Number of nodes in the new layer.
    activation : function
        Activation function (e.g., tf.nn.tanh).
    keep_prob : float
        Probability of keeping a node during dropout.
    training : bool, optional
        Whether the model is in training mode.

    Returns
    -------
    tf.Tensor
        Output of the new layer with dropout applied if training.
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )

    dense = tf.keras.layers.Dense(
        units=n, activation=activation, kernel_initializer=initializer
    )

    output = dense(prev)

    if training:
        output = tf.nn.dropout(output, rate=1 - keep_prob)

    return output
