#!/usr/bin/env python3
"""
Gradient descent with L2 regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization

    Parameters
    ----------
    Y : np.ndarray
        One-hot array of shape (classes, m) with correct labels
    weights : dict
        Dictionary of weights and biases
    cache : dict
        Dictionary of forward propagation values
    alpha : float
        Learning rate
    lambtha : float
        L2 regularization parameter
    L : int
        Number of layers

    Returns
    -------
    None
        Updates weights and biases in place
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y

    for layer in range(L, 0, -1):
        A_prev = cache[f"A{layer-1}"]
        W = weights[f"W{layer}"]

        # Gradient with L2 penalty
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update
        weights[f"W{layer}"] = W - alpha * dW
        weights[f"b{layer}"] = weights[f"b{layer}"] - alpha * db

        if layer > 1:  # Hidden layer backprop
            dZ = np.matmul(W.T, dZ) * (1 - cache[f"A{layer-1}"]**2)
