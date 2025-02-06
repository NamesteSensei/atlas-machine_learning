#!/usr/bin/env python3
"""Test script for DeepNeuralNetwork"""

import numpy as np

DeepNeuralNetwork = __import__('16-deep_neural_network').DeepNeuralNetwork

# Load the dataset (ensure the dataset is in the correct path)
lib_train = np.load('../data/Binary_Train.npz')
X_3D = lib_train['X']
Y = lib_train['Y']
# Reshape X to (nx, m)
X = X_3D.reshape((X_3D.shape[0], -1)).T

# Set a seed for reproducibility
np.random.seed(0)

# Initialize Deep Neural Network
nx = X.shape[0]
layers = [5, 3, 1]
deep = DeepNeuralNetwork(nx, layers)

# Output the initialized attributes
print("Cache:", deep.cache)
print("\nWeights:")
for key in sorted(deep.weights.keys()):
    print(f"{key}:", deep.weights[key])

print("\nNumber of layers:", deep.L)

# Modify and print L to show it's a public attribute
deep.L = 10
print("Modified number of layers:", deep.L)
