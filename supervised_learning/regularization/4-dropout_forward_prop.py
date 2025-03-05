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

    Returns: a dictionary containing the outputs of each layer and the dropout
             mask used on each layer (if applicable).
    """
    cache = {"A0": X}  # Store input as A0
    A_prev = X

    for layer_index in range(1, L + 1):
        W = weights["W" + str(layer_index)]
        b = weights["b" + str(layer_index)]
        Z = np.matmul(W, A_prev) + b  # Linear step

        if layer_index != L:
            # Apply tanh activation for hidden layers
            A = np.tanh(Z)
            # Apply dropout mask
            D = np.random.rand(*A.shape) < keep_prob
            A = A * D / keep_prob  # Scale activation during training
            cache["D" + str(layer_index)] = D  # Store dropout mask
        else:
            # Apply softmax activation for the output layer
            exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

        cache["A" + str(layer_index)] = A
        A_prev = A

    return cache
