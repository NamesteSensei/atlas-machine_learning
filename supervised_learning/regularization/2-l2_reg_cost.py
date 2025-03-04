#!/usr/bin/env python3
import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Calculates the total cost of a model with L2 regularization.

    Args:
        cost: tensor containing the original cost of the network.
        model: Keras model with L2 regularization in layers.

    Returns:
        Tensor containing the total cost including L2 regularization.
    """
    reg_cost = tf.add_n(model.losses)
    total_cost = cost + reg_cost
    return total_cost
