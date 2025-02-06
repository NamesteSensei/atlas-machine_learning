#!/usr/bin/env python3
"""
Test script for DeepNeuralNetwork class.
Loads dataset, initializes, and prints attributes.
"""

import numpy as np
Deep = __import__('16-deep_neural_network').DeepNeuralNetwork

# Load training dataset
lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

# Initialize deep neural network
np.random.seed(0)
deep = Deep(X.shape[0], [5, 3, 1])

# Print attributes
print(deep.cache)  # Should print an empty dictionary
print(deep.weights)  # Should print initialized weights & biases
print(deep.L)  # Should print the number of layers
