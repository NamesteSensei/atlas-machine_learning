#!/usr/bin/env python3
"""
Module for implementing learning rate decay using inverse time decay.
"""

import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay.

    Parameters:
    alpha (float): Initial learning rate.
    decay_rate (float): Determines how fast learning rate decays.
    global_step (int): Number of gradient descent steps taken.
    decay_step (int): Steps after which decay occurs.

    Returns:
    float: Updated learning rate.
    """
    return alpha / (1 + decay_rate * (global_step // decay_step))
