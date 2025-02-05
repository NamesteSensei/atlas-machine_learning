#!/usr/bin/env python3

import numpy as np

NN = __import__('9-neural_network').NeuralNetwork

# Load training data
lib_train = np.load('../data/train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

# Initialize the NeuralNetwork
np.random.seed(0)
nn = NN(X.shape[0], 3)

# Print values using getter methods
print(nn.W1)  # Should print weights of hidden layer
print(nn.b1)  # Should print biases of hidden layer
print(nn.W2)  # Should print weights of output layer
print(nn.b2)  # Should print biases of output layer
print(nn.A1)  # Should print activation of hidden layer (should be 0)
print(nn.A2)  # Should print activation of output layer (should be 0)

# Test attempting to modify private attribute (should raise an error)
try:
    nn.A1 = 10
except AttributeError as e:
    print(e)  # Should print an error message since A1 is private
