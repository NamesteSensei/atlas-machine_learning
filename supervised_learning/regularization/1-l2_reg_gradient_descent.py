#!/usr/bin/env python3
"""
Module for updating weights and biases of a neural network using
gradient descent with L2 regularization.
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases of a neural network using gradient descent
    with L2 regularization.

    Args:
        Y (np.ndarray): Shape (classes, m), one-hot labels.
        weights (dict): Weights and biases of the neural network.
        cache (dict): Outputs of each layer of the neural network.
        alpha (float): Learning rate.
        lambtha (float): L2 regularization parameter.
        L (int): Number of layers in the network.

    Returns:
        None: Updates weights and biases in place.
    """
    m = Y.shape[1]
    dz = cache["A" + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(i - 1)]
        W = weights["W" + str(i)]
        b = weights["b" + str(i)]

        # Compute gradients with L2 regularization
        dw = (np.matmul(dz, A_prev.T) / m) + (lambtha / m) * W
        db = np.sum(dz, axis=1, keepdims=True) / m

        if i > 1:
            # Apply tanh derivative for hidden layers
            dz = np.matmul(W.T, dz) * (1 - A_prev ** 2)

        # Update weights and biases
        weights["W" + str(i)] = W - alpha * dw
        weights["b" + str(i)] = b - alpha * db
