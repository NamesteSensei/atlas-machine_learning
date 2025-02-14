#!/usr/bin/env python3
"""Test file for training a neural network classifier."""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

train = __import__('6-train').train


def one_hot(Y, classes):
    """Converts an array to a one-hot matrix."""
    one_hot_matrix = np.zeros((Y.shape[0], classes))
    one_hot_matrix[np.arange(Y.shape[0]), Y] = 1
    return one_hot_matrix


if __name__ == '__main__':
    # Load the MNIST data
    lib = np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
    Y_train_oh = one_hot(Y_train, 10)

    X_valid_3D = lib['X_valid']
    Y_valid = lib['Y_valid']
    X_valid = X_valid_3D.reshape((X_valid_3D.shape[0], -1))
    Y_valid_oh = one_hot(Y_valid, 10)

    # Define the network structure and training parameters
    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]
    alpha = 0.01
    iterations = 1000

    tf.set_random_seed(0)
    save_path = train(X_train, Y_train_oh, X_valid, Y_valid_oh,
                      layer_sizes, activations, alpha, iterations,
                      save_path="./model.ckpt")
    print("Model saved in path: {}".format(save_path))
