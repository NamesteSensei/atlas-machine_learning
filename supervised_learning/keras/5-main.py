#!/usr/bin/env python3
"""
Main file for testing 5-train.py
"""

# Force Seed - fix for Keras
SEED = 8

import os
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS']= '0'
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
train_model = __import__('5-train').train_model

if __name__ == '__main__':
    datasets = np.load('MNIST.npz')
    X_train = datasets['X_train'].reshape(-1, 784)
    Y_train = one_hot(datasets['Y_train'])
    X_valid = datasets['X_valid'].reshape(-1, 784)
    Y_valid = one_hot(datasets['Y_valid'])

    # Model parameters
    lambtha = 0.0001
    keep_prob = 0.95
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999

    # Build and optimize model
    network = build_model(784, [256, 256, 10], 
                          ['relu', 'relu', 'softmax'], 
                          lambtha, keep_prob)
    optimize_model(network, alpha, beta1, beta2)

    # Training parameters
    batch_size = 64
    epochs = 5

    # Train the model with validation data
    train_model(network, X_train, Y_train, batch_size, epochs, 
                validation_data=(X_valid, Y_valid))
