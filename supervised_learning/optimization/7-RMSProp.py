#!/usr/bin/env python3
"""
Module for updating variables using RMSProp optimization algorithm.

RMSProp adapts the learning rate for each parameter using
a moving average of squared gradients.
"""


import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm.

    Parameters:
    alpha (float): Learning rate.
    beta2 (float): RMSProp decay factor.
    epsilon (float): Small value to avoid division by zero.
    var (numpy.ndarray): Variable to be updated.
    grad (numpy.ndarray): Gradient of the variable.
    s (numpy.ndarray): Previous second moment estimate.

    Returns:
    tuple: (updated variable, updated second moment estimate)
    """
    s = beta2 * s + (1 - beta2) * grad ** 2  # Compute second moment estimate
    var = var - alpha * (grad / (np.sqrt(s) + epsilon))  # Apply RMSProp update
    return var, s
