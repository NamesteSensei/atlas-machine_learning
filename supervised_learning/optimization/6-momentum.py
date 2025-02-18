#!/usr/bin/env python3
"""
Module for setting up Gradient Descent with Momentum in TensorFlow.

This function returns a TensorFlow optimizer configured with momentum.
"""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    Creates the Gradient Descent optimizer with Momentum.

    Parameters:
    alpha (float): Learning rate.
    beta1 (float): Momentum weight.

    Returns:
    tf.keras.optimizers.SGD: Optimizer instance with momentum.
    """
    return tf.keras.optimizers.SGD(learning_rate=alpha, momentum=beta1)
