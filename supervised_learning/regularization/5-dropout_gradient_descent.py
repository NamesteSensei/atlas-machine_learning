#!/usr/bin/env python3
"""
5-dropout_gradient_descent.py
Updates the weights of a neural network with Dropout regularization
using gradient descent.
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights of a neural network with Dropout regularization
    using gradient descent.

    Y: one-hot numpy.ndarray of shape (classes, m) containing correct labels
    weights: dictionary of the weights and biases of the neural network
    cache: dictionary containing outputs and dropout masks of each layer
    alpha: learning rate
    keep_prob: probability that a node will be kept
    L: number of layers in the network

    Updates weights in place.
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db

        if i > 1:
            dA = np.matmul(W.T, dZ) * (1 - A_prev ** 2)
            D = cache['D' + str(i - 1)]
            dA *= D
            dA /= keep_prob
            dZ = dA
