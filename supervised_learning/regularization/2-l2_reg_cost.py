#!/usr/bin/env python3
"""
Module that calculates the cost of a Keras model
with L2 regularization
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the cost of a neural network with L2 regularization

    Parameters
    ----------
    cost : tf.Tensor
        Cost of the network without L2 regularization
    model : tf.keras.Model
        A compiled Keras model that may include L2 regularization

    Returns
    -------
    tf.Tensor
        A tensor containing the total cost for each layer,
        accounting for L2 regularization
    """
    # Collect all regularization losses from the model
    reg_losses = model.losses

    # Put cost first, then each reg loss
    total_losses = [cost] + reg_losses

    # Convert to a single tensor vector
    return tf.convert_to_tensor(total_losses)
