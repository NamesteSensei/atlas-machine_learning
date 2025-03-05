#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import os
import random

# Set a consistent seed for reproducibility
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Import the L2 regularization cost function
l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost

def one_hot(Y, classes):
    """
    Converts an array of labels into a one-hot encoded matrix.
    
    Parameters:
    Y (np.ndarray): Array of labels.
    classes (int): Total number of classes.
    
    Returns:
    np.ndarray: One-hot encoded matrix.
    """
    m = Y.shape[0]
    oh = np.zeros((m, classes))
    oh[np.arange(m), Y] = 1
    return oh

# Load the dataset and prepare the data
m = 1500  # Use a fixed value for consistent testing
c = 10
lib = np.load('MNIST.npz')

# Prepare input data and labels
X = lib['X_train'][:m].reshape((m, -1))
Y = one_hot(lib['Y_train'][:m], c)

# Load the pre-trained model with L2 regularization
model_reg = tf.keras.models.load_model('model_reg.h5', compile=False)

# Generate predictions using the model
Predictions = model_reg(X)

# Calculate the base cost using categorical cross-entropy
cost = tf.keras.losses.CategoricalCrossentropy()(Y, Predictions)

# Calculate the total cost with L2 regularization
l2_cost = l2_reg_cost(cost, model_reg)

# Output the final L2 regularized cost
print(l2_cost)
