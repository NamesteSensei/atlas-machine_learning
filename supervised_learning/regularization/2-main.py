#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import random
import os

SEED = 0

# Set up a consistent environment for reproducibility
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Import the L2 regularization cost function
l2_reg_cost = __import__('2-l2_reg_cost').l2_reg_cost

def one_hot(Y, classes):
    """Convert an array to a one-hot matrix."""
    m = Y.shape[0]
    one_hot = np.zeros((m, classes))
    one_hot[np.arange(m), Y] = 1
    return one_hot

# Load the MNIST dataset
lib = np.load('MNIST.npz')
X_train_3D = lib['X_train']
Y_train = lib['Y_train']
X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
Y_train_oh = one_hot(Y_train, 10)

# Check the input data shapes
print("X_train shape:", X_train.shape)
print("Y_train_oh shape:", Y_train_oh.shape)

# Load the pre-trained model with regularization
model_reg = tf.keras.models.load_model('model_reg.h5', compile=False)

# Output model architecture
print("Model Loaded Successfully")
for layer in model_reg.layers:
    try:
        output_shape = layer.output_shape
    except AttributeError:
        output_shape = 'N/A'
    print(f"Layer: {layer.name}, Output Shape: {output_shape}")

# Make predictions using the model
predictions = model_reg.predict(X_train)
print("Predictions Shape:", predictions.shape)
print("Predictions Example (First 5):", predictions[:5])

# Calculate the categorical cross-entropy cost
cost = tf.keras.losses.CategoricalCrossentropy()(Y_train_oh, predictions)
print("Cost (without L2):", cost)

# Calculate the L2 regularized cost
l2_cost = l2_reg_cost(cost, model_reg)
print("L2 Regularized Cost:", l2_cost)
