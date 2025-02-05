#!/usr/bin/env python3
"""
Test script for the DeepNeuralNetwork class (Task 16).
Loads training data, initializes the deep neural network,
and prints important parameters.
"""

import numpy as np

# Import the DeepNeuralNetwork class
Deep = __import__('16-deep_neural_network').DeepNeuralNetwork

# Load training data
lib_train = np.load('../data/train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

# Initialize Deep Neural Network with multiple layers
np.random.seed(0)
dnn = Deep(X.shape[0], [5, 3, 1])

# Print key attributes to validate correct initialization
print(dnn.L)  # Number of layers
print(dnn.weights)  # Weights and biases
print(dnn.cache)  # Should be an empty dictionary
