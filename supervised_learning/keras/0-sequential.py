#!/usr/bin/env python3
"""Builds a Keras Sequential model."""

import tensorflow.keras as K


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
    - model (K.Sequential): The constructed model.
    """
    model = K.Sequential()
    reg = K.regularizers.L2(lambtha)

    for i, (nodes, activation) in enumerate(zip(layers, activations)):
        if i == 0:
            model.add(K.layers.Dense(nodes,
                                     activation=activation,
                                     kernel_regularizer=reg,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(nodes,
                                     activation=activation,
                                     kernel_regularizer=reg))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
