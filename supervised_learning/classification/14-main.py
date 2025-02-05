#!/usr/bin/env python3
"""
Test script for the NeuralNetwork class.
Loads training and validation data, trains the model, and evaluates accuracy.
"""

import numpy as np

# Import the NeuralNetwork class from 14-neural_network.py
NN = __import__('14-neural_network').NeuralNetwork

# Load training data
lib_train = np.load('../data/train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

# Load development (validation) data
lib_dev = np.load('../data/dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

# Initialize Neural Network
np.random.seed(0)
nn = NN(X_train.shape[0], 3)

# Train the neural network
A, cost = nn.train(X_train, Y_train, iterations=100)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))

# Evaluate model on validation data
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))
