#!/usr/bin/env python3
"""Module that creates a layer for a neural network"""

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()


def create_layer(prev, n, activation):
    """
    Creates a layer for a neural network.

    Args:
        prev: the tensor output of the previous layer
        n: number of nodes in the layer
        activation: activation function to be used in the layer

    Returns:
        The tensor output of the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=init,
        name='layer'
    )
    return layer(prev)
