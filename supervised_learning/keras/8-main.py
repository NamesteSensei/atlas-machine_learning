#!/usr/bin/env python3
"""
8-main.py
---------
Test file for Task 8: Saving and loading a trained Keras model.
"""

# Force Seed - reproducibility
SEED = 8

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import random
random.seed(SEED)
import numpy as np
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('7-train').train_model
save_model, load_model = __import__('8-save_load').save_model, __import__('8-save_load').load_model


if __name__ == '__main__':
    # Load dataset
    datasets = np.load('MNIST.npz')
    X_train = datasets['X_train']
    Y_train = datasets['Y_train']
    X_valid = datasets['X_valid']
    Y_valid = datasets['Y_valid']

    # Flatten
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_valid = X_valid.reshape(X_valid.shape[0], -1)

    # One-hot
    Y_train_oh = one_hot(Y_train)
    Y_valid_oh = one_hot(Y_valid)

    # Build + optimize
    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10],
                          ['relu', 'relu', 'softmax'],
                          lambtha, keep_prob)

    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)

    # Train with decay + early stopping
    batch_size = 64
    epochs = 5
    train_model(
        network,
        X_train,
        Y_train_oh,
        batch_size,
        epochs,
        validation_data=(X_valid, Y_valid_oh),
        early_stopping=True,
        patience=2,
        learning_rate_decay=True,
        alpha=alpha,
        decay_rate=1
    )

    # Save model with timestamp
    filename = save_model(network)
    print(f"✅ Model saved as {filename}")

    # Load model
    new_network = load_model(filename)
    print("✅ Model successfully reloaded:")
    new_network.summary()
