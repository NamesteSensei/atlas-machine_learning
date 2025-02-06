#!/usr/bin/env python3

import numpy as np

Deep = __import__('16-deep_neural_network').DeepNeuralNetwork

# Load the dataset (ensure the path to 'Binary_Train.npz' is correct)
lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
# Reshape X to have the correct dimensions (features, examples)
X = X_3D.reshape((X_3D.shape[0], -1)).T  # (nx, m)

# Set a seed for reproducibility
np.random.seed(0)

# Initialize the Deep Neural Network
deep = Deep(X.shape[0], [5, 3, 1])  # Layers with 5, 3, and 1 nodes respectively

# Print the attributes
print(deep.cache)    # Should be an empty dictionary
print(deep.weights)  # Should contain initialized weights and biases
print(deep.L)        # Number of layers

# Modify and print L to demonstrate it's a public attribute
deep.L = 10
print(deep.L)
