#!/usr/bin/env python3
"""
Test file for l2_reg_gradient_descent function.
Ensures the function updates weights and biases using gradient descent
with L2 regularization.
"""

import numpy as np
l2_reg_gradient_descent = __import__('1-l2_reg_gradient_descent')\
    .l2_reg_gradient_descent

def one_hot(Y, classes):
    """
    Converts an array of labels into a one-hot encoded matrix.
    Args:
        Y (np.ndarray): Array of shape (m,) with label indices.
        classes (int): Number of classes.
    Returns:
        np.ndarray: One-hot encoded matrix of shape (classes, m).
    """
    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    one_hot[Y, np.arange(m)] = 1
    return one_hot

if __name__ == '__main__':
    lib = np.load('MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T
    Y_train_oh = one_hot(Y_train, 10)

    np.random.seed(0)

    weights = {
        'W1': np.random.randn(256, 784),
        'b1': np.zeros((256, 1)),
        'W2': np.random.randn(128, 256),
        'b2': np.zeros((128, 1)),
        'W3': np.random.randn(10, 128),
        'b3': np.zeros((10, 1))
    }

    # Initialize the cache dictionary properly
    cache = {}
    cache['A0'] = X_train
    cache['A1'] = np.tanh(np.matmul(weights['W1'], cache['A0']) + weights['b1'])
    cache['A2'] = np.tanh(np.matmul(weights['W2'], cache['A1']) + weights['b2'])
    Z3 = np.matmul(weights['W3'], cache['A2']) + weights['b3']
    cache['A3'] = np.exp(Z3) / np.sum(np.exp(Z3), axis=0)

    print(weights['W1'])
    l2_reg_gradient_descent(Y_train_oh, weights, cache, 0.1, 0.1, 3)
    print(weights['W1'])
