#!/usr/bin/env python3
"""
3-l2_reg_create_layer.py
Creates a neural network layer with L2 regularization.
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a neural network layer with L2 regularization.

    prev: tensor containing the output of the previous layer
    n: number of nodes the new layer should contain
    activation: activation function to use
    lambtha: L2 regularization parameter

    Returns: the output of the new layer
    """
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0, mode="fan_avg"
        ),
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )
    return layer(prev)
