#!/usr/bin/env python3

import numpy as np
dropout_forward_prop = __import__('4-dropout_forward_prop').dropout_forward_prop


def one_hot(Y, classes):
    """Convert array to one-hot"""
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
        'b3': np.zeros((10, 1)),
    }

    cache = dropout_forward_prop(X_train, weights, 3, 0.8)
    for k in sorted(cache.keys()):
        print(k, cache[k])
