#!/usr/bin/env python3
"""
1-input.py
----------
Builds a neural network using the Keras Functional API,
with L2 regularization and Dropout layers.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library (Functional API).

    Parameters
    ----------
    nx : int
        Number of input features to the network.
    layers : list of int
        List containing the number of nodes in each layer.
    activations : list of str
        List containing the activation functions used
        for each layer of the network.
    lambtha : float
        L2 regularization parameter.
    keep_prob : float
        Probability that a node will be kept for dropout.

    Returns
    -------
    model : K.Model
        The constructed Keras Functional model.
    """
    # Define input layer explicitly
    inputs = K.Input(shape=(nx,))

    # First Dense layer connected to inputs
    x = K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha)
    )(inputs)

    # Add Dropout after first Dense if more layers follow
    if len(layers) > 1:
        x = K.layers.Dropout(rate=1 - keep_prob)(x)

    # Add remaining hidden + output layers
    for i in range(1, len(layers)):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

        # Add Dropout after each hidden Dense (not last)
        if i != len(layers) - 1:
            x = K.layers.Dropout(rate=1 - keep_prob)(x)

    # Define model
    model = K.Model(inputs=inputs, outputs=x, name="model")

    return model
