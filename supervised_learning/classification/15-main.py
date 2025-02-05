#!/usr/bin/env python3
"""
Test script for NeuralNetwork training (Task 15).
Loads datasets, trains the model, evaluates it, and visualizes training cost.
"""

import numpy as np
import matplotlib.pyplot as plt

# Import NeuralNetwork class from 15-neural_network.py
NN = __import__('15-neural_network').NeuralNetwork

# Load training data
lib_train = np.load('../data/train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

# Load development data
lib_dev = np.load('../data/dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

# Initialize neural network
np.random.seed(0)
nn = NN(X_train.shape[0], 3)

# Train the model with verbose and graph enabled
A, cost = nn.train(X_train, Y_train, iterations=5000, alpha=0.05, verbose=True, graph=True, step=500)

# Evaluate on training data
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))

# Evaluate on development data
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))

# Visualize some sample predictions
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis("off")
plt.tight_layout()
plt.show()
