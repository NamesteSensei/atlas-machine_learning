#!/usr/bin/env python3
"""
Task 5: Gradient Descent with Dropout
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.
    """
    m = Y.shape[1]
    dZ = cache[f"A{L}"] - Y  # derivative for softmax output

    for layer in range(L, 0, -1):
        A_prev = cache[f"A{layer-1}"]
        W = weights[f"W{layer}"]

        # Gradient for weights and biases
        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # Update in place
        weights[f"W{layer}"] -= alpha * dW
        weights[f"b{layer}"] -= alpha * db

        if layer > 1:
            # Backpropagate error to previous layer
            dA = np.matmul(W.T, dZ)

            # Apply dropout mask & rescale
            dA *= cache[f"D{layer-1}"]
            dA /= keep_prob

            # Derivative of tanh
            A_prev = cache[f"A{layer-1}"]
            dZ = dA * (1 - A_prev ** 2)
