#!/usr/bin/env python3

import numpy as np
from deep_neural_network import DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = DeepNeuralNetwork(X.shape[0], [5, 3, 1])

# Checking private attribute behavior
print(deep.cache)  # Expected: {}
print(deep.weights)  # Expected: Dictionary with W1, b1, W2, b2, W3, b3
print(deep.L)  # Expected: 3

# Testing forward propagation
A, cache = deep.forward_prop(X)
print(A)  # Expected: Output of last layer (probabilities)
print(cache.keys())  # Expected: Contains A0, A1, A2, A3

# Attempting to modify private attribute
try:
    deep.L = 10
except AttributeError as e:
    print(e)  # Expected: can't set attribute
