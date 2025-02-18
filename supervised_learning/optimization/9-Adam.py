#!/usr/bin/env python3
"""
Module for updating variables using the Adam optimization algorithm.

Adam combines Momentum and RMSProp to achieve efficient learning.
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable using the Adam optimization algorithm.

    Parameters:
    alpha (float): Learning rate.
    beta1 (float): Momentum decay factor.
    beta2 (float): RMSProp decay factor.
    epsilon (float): Small number to avoid division by zero.
    var (numpy.ndarray): Variable to be updated.
    grad (numpy.ndarray): Gradient of the variable.
    v (numpy.ndarray): Previous first moment estimate.
    s (numpy.ndarray): Previous second moment estimate.
    t (int): Time step (for bias correction).

    Returns:
    tuple: (updated variable, updated first moment, updated second moment)
    """
    # Update first moment estimate (Momentum)
    v = beta1 * v + (1 - beta1) * grad

    # Update second moment estimate (RMSProp)
    s = beta2 * s + (1 - beta2) * grad ** 2

    # Bias correction
    v_corrected = v / (1 - beta1 ** t)
    s_corrected = s / (1 - beta2 ** t)

    # Adam update step
    var = var - alpha * (v_corrected / (np.sqrt(s_corrected) + epsilon))

    return var, v, s
