#!/usr/bin/env python3
"""
Computes L2-regularized cost per layer for a Keras model.
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    Adds each layer's L2 loss to the cost individually.

    Args:
        cost: Tensor with base loss (no regularization)
        model: Keras model with L2-regularized layers

    Returns:
        Tensor: total cost per layer (base + L2 per-layer)
    """
    return tf.convert_to_tensor(
        [cost + loss for loss in model.losses]
    )
