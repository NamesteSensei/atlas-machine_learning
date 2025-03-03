#!/usr/bin/env python3

import numpy as np
l2_reg_gradient_descent = __import__('1-l2_reg_gradient_descent').l2_reg_gradient_descent


def one_hot(Y, classes):
    """Convert an array to a one-hot matrix."""
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

    # Initialize weights and biases
    weights = {
        'W1': np.random.randn(256, 784),
        'b1': np.zeros((256, 1)),
        'W2': np.random.randn(128, 256),
        'b2': np.zeros((128, 1)),
        'W3': np.random.randn(10, 128),
        'b3': np.zeros((10, 1))
    }

    # Forward propagation with tanh and softmax activations
    cache = {}
    cache['A0'] = X_train
    cache['A1'] = np.tanh(np.matmul(weights['W1'], cache['A0']) + weights['b1'])
    cache['A2'] = np.tanh(np.matmul(weights['W2'], cache['A1']) + weights['b2'])
    Z3 = np.matmul(weights['W3'], cache['A2']) + weights['b3']
    cache['A3'] = np.exp(Z3) / np.sum(np.exp(Z3), axis=0)

    # Display weights before update
    print("Weights before update (W1):")
    print(weights['W1'])

    # Perform gradient descent with L2 regularization
    l2_reg_gradient_descent(Y_train_oh, weights, cache, alpha=0.1, lambtha=0.1, L=3)

    # Display weights after update
    print("\nWeights after update (W1):")
    print(weights['W1'])
