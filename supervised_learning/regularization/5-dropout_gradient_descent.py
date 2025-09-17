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
    dZ = cache[f"A{L}"] - Y   # output error (softmax + cross-entropy)

    for layer in reversed(range(1, L + 1)):
        A_prev = cache[f"A{layer-1}"]
        W_curr = weights[f"W{layer}"]  # keep original for backprop

        # gradients
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # update weights in place
        weights[f"W{layer}"] = W_curr - alpha * dW
        weights[f"b{layer}"] = weights[f"b{layer}"] - alpha * db

        if layer > 1:
            # backpropagate error through tanh
            dA_prev = np.matmul(W_curr.T, dZ)
            dZ = d
