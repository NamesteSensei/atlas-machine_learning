#!/usr/bin/env python3
import numpy as np

def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases of a neural network using gradient
    descent with L2 regularization.

    Parameters:
    - Y: One-hot numpy.ndarray of shape (classes, m) with the correct labels
    - weights: Dictionary of the weights and biases
    - cache: Dictionary of outputs of each layer
    - alpha: Learning rate
    - lambtha: L2 regularization parameter
    - L: Number of layers in the network
    """

    m = Y.shape[1]

    for i in reversed(range(1, L + 1)):
        A_prev = cache[f'A{i-1}']
        A = cache[f'A{i}']
        
        if i == L:
            dZ = A - Y
        else:
            dZ = (1 - A ** 2) * np.dot(weights[f'W{i+1}'].T, dZ)

        dW = (np.dot(dZ, A_prev.T) / m) + (lambtha / m) * weights[f'W{i}']
        db = np.sum(dZ, axis=1, keepdims=True) / m

        weights[f'W{i}'] -= alpha * dW
        weights[f'b{i}'] -= alpha * db
