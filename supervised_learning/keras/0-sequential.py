#!/usr/bin/env python3
"""Builds a Keras Sequential model."""

import tensorflow as tf


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a Keras Sequential model with specified parameters.

    Parameters:
    - nx (int): Number of input features.
    - layers (list): Number of nodes in each layer.
    - activations (list): Activation functions for each layer.
    - lambtha (float): L2 regularization parameter.
    - keep_prob (float): Probability for dropout (1 - dropout rate).

    Returns:
    - model (tf.keras.Sequential): The constructed model.
    """
    model = tf.keras.Sequential()
    reg = tf.keras.regularizers.L2(lambtha)

    for i, (nodes, activation) in enumerate(zip(layers, activations)):
        if i == 0:
            model.add(tf.keras.layers.Dense(nodes,
                                            activation=activation,
                                            kernel_regularizer=reg,
                                            input_shape=(nx,)))
        else:
            model.add(tf.keras.layers.Dense(nodes,
                                            activation=activation,
                                            kernel_regularizer=reg))
        if i < len(layers) - 1:
            model.add(tf.keras.layers.Dropout(1 - keep_prob))

    return model
