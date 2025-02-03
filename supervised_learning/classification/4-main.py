#!/usr/bin/env python3

import numpy as np

Neuron = __import__('4-neuron').Neuron

# Load dataset using the correct path
lib_train = np.load('/home/christopher/data/train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T  # Reshape the input data properly

np.random.seed(0)
neuron = Neuron(X.shape[0])  # Initialize the neuron

# Evaluate the neuron’s performance on the dataset
A, cost = neuron.evaluate(X, Y)
print(A)     # Print predictions
print(cost)  # Print cost function result
