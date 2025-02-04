#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

Neuron = __import__('7-neuron').Neuron

# Load training data (update the filename)
lib_train = np.load('../data/train.npz')  # Changed from Binary_Train.npz
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

# Load development data (update the filename)
lib_dev = np.load('../data/dev.npz')  # Changed from Binary_Dev.npz
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

# Initialize the neuron
np.random.seed(0)
neuron = Neuron(X_train.shape[0])

# Train the neuron
A, cost = neuron.train(X_train, Y_train, iterations=3000, alpha=0.05, verbose=True, graph=True, step=100)

# Evaluate performance on training data
train_accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {:.2f}%".format(train_accuracy))

# Evaluate performance on development data
A, cost = neuron.evaluate(X_dev, Y_dev)
dev_accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {:.2f}%".format(dev_accuracy))

# Visualize predictions
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_dev_3D[i])
    plt.title(str(A[0, i]))
    plt.axis('off')
plt.tight_layout()
plt.show()
