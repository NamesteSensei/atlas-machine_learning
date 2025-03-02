#!/usr/bin/env python3
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization.

    Parameters:
    cost (float): Initial cost of the network without regularization
    lambtha (float): Regularization parameter (lambda)
    weights (dict): Dictionary of weights and biases of the neural network
    L (int): Number of layers in the network
    m (int): Number of data points used

    Returns:
    float: The cost of the network including L2 regularization
    """
    l2_reg_sum = 0
    # Calculate the L2 regularization term by summing squares of weights
    for i in range(1, L + 1):
        l2_reg_sum += np.sum(np.square(weights[f'W{i}']))

    # Add the L2 regularization term to the original cost
    l2_cost = cost + (lambtha / (2 * m)) * l2_reg_sum
    return l2_cost
