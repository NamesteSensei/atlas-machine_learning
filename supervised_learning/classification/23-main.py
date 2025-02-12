#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

Deep = __import__('23-deep_neural_network').DeepNeuralNetwork

# Load training and development data
lib_train = np.load('../data/Binary_Train.npz')
X_train_3D, Y_train = lib_train['X'], lib_train['Y']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1)).T

lib_dev = np.load('../data/Binary_Dev.npz')
X_dev_3D, Y_dev = lib_dev['X'], lib_dev['Y']
X_dev = X_dev_3D.reshape((X_dev_3D.shape[0], -1)).T

np.random.seed(0)
deep = Deep(X_train.shape[0], [5, 3, 1])
A, cost = deep.train(X_train, Y_train)
accuracy = np.sum(A == Y_train) / Y_train.shape[1] * 100
print("Train cost:", cost)
print("Train accuracy: {}%".format(accuracy))

A, cost = deep.evaluate(X_dev, Y_dev)
accuracy = np.sum(A == Y_dev) / Y_dev.shape[1] * 100
print("Dev cost:", cost)
print("Dev accuracy: {}%".format(accuracy))

# Plot a grid of development images with predictions
fig = plt.figure(figsize=(10, 10))
for i in range(100):
    ax = fig.add_subplot(10, 10, i + 1)
    ax.imshow(X_dev_3D[i])
    ax.set_title(A[0, i])
    ax.axis('off')
plt.tight_layout()
plt.show()
