#!/usr/bin/env python3
"""
Test file for l2_reg_cost function.
Ensures that the function calculates the cost of a Keras model
with L2 regularization correctly.
"""

import numpy as np
import tensorflow as tf
import os
import random

l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost

def one_hot(Y, classes):
    """
    Converts an array of labels into a one-hot encoded matrix.

    Args:
        Y (np.ndarray): Array of shape (m,) with label indices.
        classes (int): Number of classes.

    Returns:
        np.ndarray: One-hot encoded matrix of shape (m, classes).
    """
    m = Y.shape[0]
    oh = np.zeros((m, classes))
    oh[np.arange(m), Y] = 1
    return oh

if __name__ == '__main__':
    # Set random seed for reproducibility
    SEED = 0
    os.environ['PYTHONHASHSEED'] = str(SEED)
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    # Generate synthetic data
    m = np.random.randint(1000, 2000)  # Number of data points
    c = 10  # Number of classes
    lib = np.load('MNIST.npz')

    X = lib['X_train'][:m].reshape((m, -1))  # Reshape input data
    Y = one_hot(lib['Y_train'][:m], c)  # One-hot encode labels

    # Load the pre-trained model with L2 regularization
    model_reg = tf.keras.models.load_model('model_reg.h5', compile=False)

    # Get model predictions and compute the initial cost
    Predictions = model_reg(X)
    cost = tf.keras.losses.CategoricalCrossentropy()(Y, Predictions)

    # Calculate L2 regularized cost
    l2_cost = l2_reg_cost(cost, model_reg)
    print(l2_cost)
