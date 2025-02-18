#!/usr/bin/env python3
"""
Module for implementing learning rate decay in TensorFlow.
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    Creates a learning rate decay operation using inverse time decay.

    Parameters:
    alpha (float): Initial learning rate.
    decay_rate (float): Rate at which learning rate decays.
    decay_step (int): Steps after which decay occurs.

    Returns:
    tf.keras.optimizers.schedules.InverseTimeDecay: Learning rate schedule.
    """
    return tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
