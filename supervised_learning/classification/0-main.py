#!/usr/bin/env python3
import numpy as np
from deep_neural_network import DeepNeuralNetwork

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
deep = DeepNeuralNetwork(X.shape[0], [5, 3, 1])
A, cost = deep.evaluate(X, Y)
print(A)
print(cost)
