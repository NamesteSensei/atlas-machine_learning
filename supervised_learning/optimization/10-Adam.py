#!/usr/bin/env python3
"""
Module for setting up the Adam optimization algorithm in TensorFlow.

Adam is an adaptive learning rate optimization algorithm that combines
Momentum and RMSProp for better convergence.
"""

import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """
    Creates the Adam optimizer.

    Parameters:
    alpha (float): Learning rate.
    beta1 (float): Weight for the first moment estimate (Momentum).
    beta2 (float): Weight for the second moment estimate (RMSProp).
    epsilon (float): Small number to avoid division by zero.

    Returns:
    tf.keras.optimizers.Adam: Optimizer instance.
    """
    return tf.keras.optimizers.Adam(
        learning_rate=alpha, beta_1=beta1, beta_2=beta2, epsilon=epsilon
    )
