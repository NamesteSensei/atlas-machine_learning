#!/usr/bin/env python3

import numpy as np

# Import the NeuralNetwork class from 13-neural_network.py
NN = __import__('13-neural_network').NeuralNetwork

# Load training data
lib_train = np.load('../data/train.npz')  # Ensure this file exists
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T  # Reshape input data

# Initialize the Neural Network with input features and 3 hidden nodes
np.random.seed(0)
nn = NN(X.shape[0], 3)

# Perform forward propagation
A1, A2 = nn.forward_prop(X)

# Perform gradient descent
nn.gradient_descent(X, Y, A1, A2, 0.5)

# Print updated weights & biases
print(nn.W1)
print(nn.b1)
print(nn.W2)
print(nn.b2)
