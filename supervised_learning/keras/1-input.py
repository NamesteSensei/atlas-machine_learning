#!/usr/bin/env python3
"""
Module: 1-input
Builds a neural network using the Keras Functional API with L2 regularization
and dropout. This module defines the build_model function.
"""

import tensorflow.keras as K

def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using the Keras Functional API.

    Parameters:
    - nx (int): Number of input features.
    - layers (list of int): Number of nodes in each layer.
    - activations (list of str): Activation functions for each layer.
    - lambtha (float): L2 regularization parameter.
    - keep_prob (float): Probability of keeping a node during dropout.

    Returns:
    - keras.Model: The compiled Keras model.
    """
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i, (nodes, activation) in enumerate(zip(layers, activations)):
        x = K.layers.Dense(
            units=nodes,
            activation=activation,
            kernel_regularizer=K.regularizers.L2(lambtha)
        )(x)
        if i < len(layers) - 1:
            x = K.layers.Dropout(1 - keep_prob)(x)

    model = K.Model(inputs=inputs, outputs=x)

    return model
