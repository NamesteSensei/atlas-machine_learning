#!/usr/bin/env python3
"""
Forward propagation with Dropout regularization.
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conducts forward propagation using Dropout.

    Args:
        X: input data (nx, m)
        weights: dictionary of weights and biases
        L: number of layers
        keep_prob: probability of keeping a node

    Returns:
        Dictionary of outputs and dropout masks
    """
    cache = {'A0': X}

    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A_prev = cache['A' + str(i - 1)]

        Z = np.matmul(W, A_prev) + b

        if i != L:
            A = np.tanh(Z)
            D = np.random.binomial(1, keep_prob, size=A.shape)
            A *= D
            A /= keep_prob
            cache['D' + str(i)] = D
        else:
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        cache['A' + str(i)] = A

    return cache
