#!/usr/bin/env python3
"""
Updates weights and biases using gradient descent with L2 regularization.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization.

    Parameters:
    Y (numpy.ndarray): One-hot encoded labels of shape (classes, m)
    weights (dict): Weights and biases of the neural network
    cache (dict): Outputs of each layer of the network
    alpha (float): Learning rate
    lambtha (float): L2 regularization parameter
    L (int): Number of layers in the neural network

    Returns:
    None: Updates weights and biases in place
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y  # Gradient of the cost w.r.t. output

    for i in reversed(range(1, L + 1)):
        A_prev = cache[f"A{i-1}"]
        W = weights[f"W{i}"]
        b = weights[f"b{i}"]

        # Gradient with L2 regularization
        dW = (np.matmul(dZ, A_prev.T) / m) + ((lambtha / m) * W)
        db = np.sum(dZ, axis=1, keepdims=True) / m

        # Update weights and biases
        weights[f"W{i}"] -= alpha * dW
        weights[f"b{i}"] -= alpha * db

        # Compute dZ for the previous layer
        if i > 1:
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - np.square(A_prev))  # Derivative of tanh
