#!/usr/bin/env python3
"""
Gradient descent with Dropout regularization.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates weights of neural net using dropout + gradient descent.

    Args:
        Y: one-hot labels, shape (classes, m)
        weights: dict of weights and biases
        cache: dict of activations and dropout masks
        alpha: learning rate
        keep_prob: probability to keep a node
        L: number of layers

    Returns:
        None. Updates weights in-place.
    """
    m = Y.shape[1]
    dz = cache['A' + str(L)] - Y

    for i in reversed(range(1, L + 1)):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]

        dw = (1 / m) * np.matmul(dz, A_prev.T)
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)

        weights['W' + str(i)] -= alpha * dw
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            da = np.matmul(W.T, dz)
            dz = da * (1 - cache['A' + str(i - 1)] ** 2)
            dz *= cache['D' + str(i - 1)]
            dz /= keep_prob
