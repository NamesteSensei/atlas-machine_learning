#!/usr/bin/env python3

import numpy as np

Neuron = __import__('3-neuron').Neuron

# Load training data
lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T  # Reshape the dataset for training

# Initialize the neuron
np.random.seed(0)
neuron = Neuron(X.shape[0])

# Perform forward propagation
A = neuron.forward_prop(X)

# Compute cost
cost = neuron.cost(Y, A)

# Print cost
print(cost)
