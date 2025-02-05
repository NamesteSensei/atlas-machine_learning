#!/usr/bin/env python3

import numpy as np

# Import the NeuralNetwork class from 10-neural_network.py
NN = __import__('10-neural_network').NeuralNetwork

# Load training data
lib_train = np.load('../data/train.npz')  # Make sure this file exists
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T  # Reshape the input data

# Initialize the Neural Network with input features and 3 hidden nodes
np.random.seed(0)
nn = NN(X.shape[0], 3)

# Manually modify biases to ensure deterministic output
nn._NeuralNetwork__b1 = np.ones((3, 1))
nn._NeuralNetwork__b2 = 1

# Run forward propagation
A1, A2 = nn.forward_prop(X)

# Print results to verify output
if A1 is nn.A1:
    print(A1)

if A2 is nn.A2:
    print(A2)
