#!/usr/bin/env python3
"""
Module that updates the weights and biases of a neural network
using gradient descent with L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates the weights and biases of a neural network using
    gradient descent with L2 regularization

    Parameters
    ----------
    Y : np.ndarray (classes, m)
        One-hot labels
    weights : dict
        Dictionary of weights and biases
    cache : dict
        Dictionary of activations
    alpha : float
        Learning rate
    lambtha : float
        L2 regularization parameter
    L : int
        Number of layers

    Notes
    -----
    Updates `weights` in place
    """
    m = Y.shape[1]

    for l in reversed(range(1, L + 1)):
        A_curr = cache[f"A{l}"]
        A_prev = cache[f"A{l-1}"]

        if l == L:  # softmax layer
            dZ = A_curr - Y
        else:  # tanh layers
            dZ = (np.matmul(weights[f"W{l+1}"].T, dZ)) * (1 - A_curr ** 2)

        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * weights[f"W{l}"]
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update
        weights[f"W{l}"] -= alpha * dW
        weights[f"b{l}"] -= alpha * db
