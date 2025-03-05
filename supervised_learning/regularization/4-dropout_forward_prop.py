#!/usr/bin/env python3
"""
4-dropout_forward_prop.py
Conducts forward propagation using dropout.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    X: numpy.ndarray of shape (nx, m) containing the input data
    weights: dictionary of the weights and biases of the neural network
    L: number of layers in the network
    keep_prob: probability that a node will be kept

    Returns: a dictionary containing the outputs of each layer and the
             dropout mask used on each layer (if applicable).
    """
    cache = {'A0': X}
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        Z = np.matmul(W, cache['A' + str(i - 1)]) + b

        if i != L:
            A = np.tanh(Z)
            D = np.random.rand(*A.shape) < keep_prob
            A = (A * D) / keep_prob
            cache['D' + str(i)] = D
        else:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        cache['A' + str(i)] = A
    return cache
