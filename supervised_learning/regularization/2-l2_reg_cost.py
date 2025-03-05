#!/usr/bin/env python3
"""
2-l2_reg_cost.py
Calculates the total cost of a neural network with L2 regularization.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    cost: a tensor containing the cost of the network without L2 regularization
    model: a Keras model that includes layers with L2 regularization

    Returns: a tensor containing the total cost for each layer of the network,
             accounting for L2 regularization.
    """
    l2_costs = []  # Initialize a list to hold per-layer costs
    for layer in model.layers:
        # Check if the layer has a kernel and a kernel_regularizer
        if hasattr(layer, "kernel") and layer.kernel_regularizer is not None:
            # Compute the regularization cost for the layer's kernel
            reg_cost = layer.kernel_regularizer(layer.kernel)
            # Add the regularization cost proportionally to the base cost
            layer_cost = cost + reg_cost
            l2_costs.append(layer_cost)

    # Convert the list to a tensor to match expected output format
    return tf.convert_to_tensor(l2_costs)
