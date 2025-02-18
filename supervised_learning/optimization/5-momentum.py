#!/usr/bin/env python3
"""
Module for updating variables using Gradient Descent with Momentum.

This optimization method improves convergence speed and reduces oscillations.
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using Gradient Descent with Momentum.

    Parameters:
    alpha (float): Learning rate.
    beta1 (float): Momentum coefficient.
    var (numpy.ndarray): Variable to be updated.
    grad (numpy.ndarray): Gradient of the variable.
    v (numpy.ndarray): Previous velocity (momentum term).

    Returns:
    tuple: (updated variable, updated velocity)
    """
    v = beta1 * v + (1 - beta1) * grad  # Compute velocity
    var = var - alpha * v  # Update variable
    return var, v
