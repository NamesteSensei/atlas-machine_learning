#!/usr/bin/env python3
"""Test file for evaluating the network using the evaluate.chpt checkpoint."""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

evaluate = __import__('7-evaluate').evaluate


def one_hot(Y, classes):
    """Convert an array to a one-hot matrix."""
    one_hot_matrix = np.zeros((Y.shape[0], classes))
    one_hot_matrix[np.arange(Y.shape[0]), Y] = 1
    return one_hot_matrix


if __name__ == "__main__":
    # Set NumPy print options to match expected output
    np.set_printoptions(precision=3, suppress=True)
    
    # Load the MNIST data using the uppercase filename
    data = np.load("../data/MNIST.npz")
    X_test = data["X_test"]
    X_test = X_test.reshape((X_test.shape[0], -1))
    Y_test = data["Y_test"]
    Y_test_oh = one_hot(Y_test, 10)

    # Evaluate the network using the saved model from evaluate.chpt
    pred, _, _ = evaluate(X_test, Y_test_oh, "evaluate.chpt")
    
    # Print the network's prediction array in the required format
    print(pred)
