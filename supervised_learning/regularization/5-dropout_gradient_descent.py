#!/usr/bin/env python3
"""
Gradient Descent with Dropout Regularization
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Parameters
    ----------
    Y : np.ndarray
        One-hot array of shape (classes, m) with true labels.
    weights : dict
        Dictionary of weights and biases to update in place.
    cache : dict
        Dictionary of forward activations and dropout masks.
    alpha : float
        Learning rate.
    keep_prob : float
        Probability of keeping a node during dropout.
    L : int
        Number of layers in the network.
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y   # derivative at output (softmax + cross-entropy)

    for l in reversed(range(1, L + 1)):
        A_prev = cache[f"A{l-1}"]
        W = weights[f"W{l}"]

        # compute gradients
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # update parameters
        weights[f"W{l}"] = W - alpha * dW
        weights[f"b{l}"] = weights[f"b{l}"] - alpha * db

        if l > 1:
            # propagate error backwards through tanh
            dA_prev = np.matmul(W.T, dZ)
            dZ = dA_prev * (1 - (A_prev ** 2))

            # apply dropout mask
            D_prev = cache[f"D{l-1}"]
            dZ = dZ * D_prev / keep_prob
