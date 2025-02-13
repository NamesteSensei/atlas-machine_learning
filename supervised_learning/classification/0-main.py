#!/usr/bin/env python3

import numpy as np
import importlib.util
import sys

# Load 21-deep_neural_network.py dynamically
spec = importlib.util.spec_from_file_location("deep_neural_network", "21-deep_neural_network.py")
deep_neural_network = importlib.util.module_from_spec(spec)
sys.modules["deep_neural_network"] = deep_neural_network
spec.loader.exec_module(deep_neural_network)

# Now import the DeepNeuralNetwork class
DeepNeuralNetwork = deep_neural_network.DeepNeuralNetwork

# Load dataset
lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

# Initialize and test the deep neural network
np.random.seed(0)
deep = DeepNeuralNetwork(X.shape[0], [5, 3, 1])
A, cost = deep.evaluate(X, Y)

print(A)
print(cost)
