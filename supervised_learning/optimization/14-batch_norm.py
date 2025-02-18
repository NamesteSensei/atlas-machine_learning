#!/usr/bin/env python3
"""
Batch Normalization Layer in TensorFlow.
"""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer in TensorFlow.

    Parameters:
    prev (tensor): Activated output from previous layer.
    n (int): Number of nodes in the new layer.
    activation (function): Activation function to apply.

    Returns:
    tensor: Activated output of the layer.
    """
    # Dense Layer with VarianceScaling initializer
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    dense = tf.keras.layers.Dense(units=n, kernel_initializer=init)
    Z = dense(prev)

    # Compute BatchNorm
    mean, variance = tf.nn.moments(Z, axes=[0])

    # Trainable parameters (gamma and beta)
    gamma = tf.Variable(tf.ones([n]), trainable=True)
    beta = tf.Variable(tf.zeros([n]), trainable=True)

    # Normalize the activations
    Z_norm = tf.nn.batch_normalization(Z, mean, variance, beta, gamma, 1e-7)

    # Apply activation function
    return activation(Z_norm)
