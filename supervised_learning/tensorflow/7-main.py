#!/usr/bin/env python3
"""Test file for evaluating a neural network classifier."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

evaluate = __import__('7-evaluate').evaluate


def one_hot(Y, classes):
    """Converts an array to a one-hot matrix."""
    one_hot_matrix = np.zeros((Y.shape[0], classes))
    one_hot_matrix[np.arange(Y.shape[0]), Y] = 1
    return one_hot_matrix


if __name__ == '__main__':
    # Load the MNIST dataset
    lib = np.load('../data/MNIST.npz')
    X_test_3D = lib['X_test']
    Y_test = lib['Y_test']
    X_test = X_test_3D.reshape((X_test_3D.shape[0], -1))
    Y_test_oh = one_hot(Y_test, 10)

    # Evaluate the network using the saved model
    Y_pred_oh, accuracy, cost = evaluate(X_test, Y_test_oh, './model.ckpt')
    print("Test Accuracy:", accuracy)
    print("Test Cost:", cost)

    # Convert predictions from one-hot to class labels
    Y_pred = np.argmax(Y_pred_oh, axis=1)

    # Display the first 100 test images with their predicted and true labels
    fig = plt.figure(figsize=(10, 10))
    for i in range(100):
        fig.add_subplot(10, 10, i + 1)
        plt.imshow(X_test_3D[i], cmap="gray")
        plt.title("{} : {}".format(Y_test[i], Y_pred[i]))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
