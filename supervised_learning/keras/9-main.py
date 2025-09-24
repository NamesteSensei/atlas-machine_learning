#!/usr/bin/env python3
"""
9-main.py
---------
Test file for Task 9: Saving and loading only model weights.
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
save_weights, load_weights = __import__('9-save_load').save_weights, __import__('9-save_load').load_weights


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

    # Train
    batch_size = 64
    epochs = 3
    train_model(
        network,
        X_train,
        Y_train_oh,
        batch_size,
        epochs,
        validation_data=(X_valid, Y_valid_oh)
    )

    # Save weights
    weight_file = "model_weights.h5"
    save_weights(network, weight_file)
    print(f"✅ Weights saved to {weight_file}")

    # Rebuild the same model structure
    new_network = build_model(784, [256, 256, 10],
                              ['relu', 'relu', 'softmax'],
                              lambtha, keep_prob)
    optimize_model(new_network, alpha, beta1, beta2)

    # Load weights into the new model
    load_weights(new_network, weight_file)
    print("✅ Weights loaded into new model")
    new_network.summary()
