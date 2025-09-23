#!/usr/bin/env python3
"""
0-sequential.py
---------------
Builds a neural network with Keras Sequential API,
including L2 regularization and Dropout layers.
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network with the Keras library.

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
    model : K.Sequential
        The constructed Keras Sequential model.
    """
    model = K.Sequential(name="sequential")

    # First layer with input_dim
    model.add(K.layers.Dense(
        units=layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha),
        input_dim=nx
    ))

    # Add remaining layers
    for i in range(1, len(layers)):
        model.add(K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
        ))

        # Add Dropout after each Dense except the last
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(rate=1 - keep_prob))

    return model
