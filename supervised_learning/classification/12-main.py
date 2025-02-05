#!/usr/bin/env python3

import numpy as np

# Import the NeuralNetwork class from 12-neural_network.py
NN = __import__('12-neural_network').NeuralNetwork

# Load training data
lib_train = np.load('../data/train.npz')  # Ensure this file exists
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T  # Reshape the input data

# Initialize the Neural Network with input features and 3 hidden nodes
np.random.seed(0)
nn = NN(X.shape[0], 3)

# Perform evaluation
A, cost = nn.evaluate(X, Y)

# Print results
print(A)  # Predicted labels (0 or 1)
print(cost)  # Computed cost value
