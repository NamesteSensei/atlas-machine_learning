#!/usr/bin/env python3
"""
Test script for the NeuralNetwork class (task 14).
Loads training and development datasets, trains the network, evaluates it,
and displays sample predictions.
"""

import matplotlib.pyplot as plt
import numpy as np

# Import the NeuralNetwork class from 14-neural_network.py.
NN = __import__('14-neural_network').NeuralNetwork

# Load training data from train.npz.
lib_train = np.load('../data/train.npz')
# Assumes the npz file has keys 'X' and 'Y'.
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
# Reshape the training images into column vectors.
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

# Load development data from dev.npz.
lib_dev = np.load('../data/dev.npz')
# Assumes the npz file has keys 'X' and 'Y'.
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
# Reshape the development images into column vectors.
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

# Set the random seed for reproducibility.
np.random.seed(0)
# Create a NeuralNetwork instance with input features and 3 hidden nodes.
nn = NN(X_train.shape[0], 3)

# Train the neural network for 100 iterations.
A, cost = nn.train(X_train, Y_train, iterations=100)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))

# Evaluate the network on the development data.
A, cost = nn.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))

# Visualize the first 100 development images with their predictions.
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(A[0, i])
    plt.axis("off")
plt.tight_layout()
plt.show()
