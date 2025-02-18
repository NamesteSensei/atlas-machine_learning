#!/usr/bin/env python3
"""
Module for setting up RMSProp optimization in TensorFlow.

This function returns a TensorFlow optimizer configured with RMSProp.
"""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """
    Creates the RMSProp optimizer.

    Parameters:
    alpha (float): Learning rate.
    beta2 (float): RMSProp decay factor (discounting factor).
    epsilon (float): Small number to avoid division by zero.

    Returns:
    tf.keras.optimizers.RMSprop: Optimizer instance.
    """
    return tf.keras.optimizers.RMSprop(
        learning_rate=alpha, rho=beta2, epsilon=epsilon
    )
