#!/usr/bin/env python3
"""
Module that defines l2_reg_cost
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization.

    Args:
        cost: tensor containing the base cost (e.g. categorical cross-entropy)
        model: keras model containing the layers with L2 regularization losses

    Returns:
        A tensor containing the total cost (cost + L2 regularization)
    """
    # collect all L2 regularization losses from the model
    l2_losses = model.losses

    # add base cost and all l2 losses
    total_cost = tf.add_n([cost] + l2_losses)

    return total_cost
