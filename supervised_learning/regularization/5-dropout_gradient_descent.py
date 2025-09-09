#!/usr/bin/env python3
"""
Gradient Descent with Dropout regularization.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network using
    gradient descent with Dropout regularization.

    Args:
        Y: one-hot matrix of correct labels (classes, m)
        weights: dictionary of weights and biases
        cache: dictionary of all intermediary values of the network
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers

    Returns:
        None. Updates weights in-place.
    """
    m = Y.shape[1]
    dZ = cache["A" + str(L)] - Y  # softmax derivative

    for l in reversed(range(1, L + 1)):
        A_prev = cache["A" + str(l - 1)]
        W = weights["W" + str(l)]

        dW = (1 / m) * np.matmul(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        weights["W" + str(l)] -= alpha * dW
        weights["b" + str(l)] -= alpha * db

        if l > 1:
            # Backprop through tanh + dropout
            dA_prev = np.matmul(W.T, dZ)
            D = cache["D" + str(l - 1)]
            dA_prev *= D
            dA_prev /= keep_prob

            A_prev = cache["A" + str(l - 1)]
            dZ = dA_prev * (1 - A_prev ** 2)  # tanh derivative
