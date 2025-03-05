#!/usr/bin/env python3
"""
6-dropout_create_layer.py
Creates a layer of a neural network using Dropout.
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Creates a layer of a neural network using Dropout.

    prev: tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: activation function for the new layer
    keep_prob: probability that a node will be kept
    training: boolean indicating if the model is in training mode

    Returns: the output of the new layer.
    """
    init = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode="fan_avg"
    )

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init
    )(prev)

    if training:
        layer = tf.keras.layers.Dropout(1 - keep_prob)(layer)

    return layer
