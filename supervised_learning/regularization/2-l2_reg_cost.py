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
        A tensor containing the total cost, including L2 losses
    """
    # model.losses contains all regularization losses
    l2_loss = tf.add_n(model.losses)

    # Add base cost + regularization losses
    total_cost = cost + l2_loss
    return total_cost
