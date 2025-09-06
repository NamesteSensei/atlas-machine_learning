#!/usr/bin/env python3
"""
Module that computes the cost of a neural network with L2 regularization
"""

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    Parameters
    ----------
    cost : float
        Cost of the network without L2 regularization.
    lambtha : float
        Regularization parameter (lambda).
    weights : dict
        Dictionary of the weights and biases of the neural network.
        Keys are like 'W1', 'b1', ..., 'WL', 'bL'.
    L : int
        Number of layers in the neural network.
    m : int
        Number of data points used.

    Returns
    -------
    float
        The cost of the network accounting for L2 regularization.
    """
    l2_sum = 0
    for i in range(1, L + 1):
        W = weights[f"W{i}"]
        l2_sum += np.sum(np.square(W))

    l2_cost = cost + (lambtha / (2 * m)) * l2_sum
    return l2_cost

