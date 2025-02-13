#!/usr/bin/env python3
import numpy as np
from 21-deep_neural_network import DeepNeuralNetwork

# Load dataset
lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

# Initialize Deep Neural Network
np.random.seed(0)
deep = DeepNeuralNetwork(X.shape[0], [5, 3, 1])

# Forward propagation
A, cache = deep.forward_prop(X)

# Perform one gradient descent update
deep.gradient_descent(Y, cache, 0.5)

# Print updated weights
print(deep.weights)

