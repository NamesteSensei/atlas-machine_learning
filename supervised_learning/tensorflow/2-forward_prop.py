#!/usr/bin/env python3
""" Forward propagation in a neural network """

import tensorflow.compat.v1 as tf

# Import the create_layer function from Task 1
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        x: placeholder for input data
        layer_sizes: list with the number of nodes in each layer
        activations: list of activation functions for each layer

    Returns:
        The final output tensor (y_pred)
    """
    layer = x  # Start with input layer

    for i in range(len(layer_sizes)):
        layer = create_layer(layer, layer_sizes[i], activations[i])

    return layer  # Output of the final layer (y_pred)
