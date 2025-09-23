#!/usr/bin/env python3
"""
4-main.py
---------
Test file for Task 4: Training a Keras model using mini-batch
gradient descent on the MNIST dataset.
"""

# Force Seed - fix for Keras reproducibility
SEED = 8

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

# Imports from previous tasks
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('4-train').train_model


if __name__ == '__main__':
    # Load MNIST dataset
    datasets = np.load('MNIST.npz')
    X_train = datasets['X_train']  # shape (60000, 28, 28)
    Y_train = datasets['Y_train']  # shape (60000,)

    # Flatten images from 28x28 -> 784
    X_train = X_train.reshape(X_train.shape[0], -1)

    # Convert labels to one-hot encoding
    Y_train_oh = one_hot(Y_train)

    # Build model
    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(
        784,                      # number of input features
        [256, 256, 10],           # 2 hidden layers + output layer
        ['relu', 'relu', 'softmax'],  # activations
        lambtha,
        keep_prob
    )

    # Optimize model with Adam
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)

    # Train model
    batch_size = 64
    epochs = 5
    history = train_model(
        network,
        X_train,
        Y_train_oh,
        batch_size,
        epochs
    )
