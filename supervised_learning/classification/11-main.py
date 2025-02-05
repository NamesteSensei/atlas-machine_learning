#!/usr/bin/env python3

import numpy as np

# Import the NeuralNetwork class from 11-neural_network.py
NN = __import__('11-neural_network').NeuralNetwork

# Load training data
lib_train = np.load('../data/train.npz')  # Ensure this file exists
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T  # Reshape the input data

# Initialize the Neural Network with input features and 3 hidden nodes
np.random.seed(0)
nn = NN(X.shape[0], 3)

# Perform forward propagation
_, A = nn.forward_prop(X)

# Compute and print the cost
cost = nn.cost(Y, A)
print(cost)
