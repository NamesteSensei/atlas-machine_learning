#!/usr/bin/env python3
"""
Main file for testing Task 4: Model Training
"""

# Force Seed - fix for Keras
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

# Import functions
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('4-train').train_model

if __name__ == '__main__':
    # Load and preprocess dataset
    datasets = np.load('MNIST.npz')
    X_train = datasets['X_train'].reshape(datasets['X_train'].shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train, 10)

    # Build and compile the model
    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(784, [256, 256, 10], ['relu', 'relu', 'softmax'],
                          lambtha, keep_prob)
    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)

    # Train the model
    batch_size = 64  # 64 tricks per session
    epochs = 5       # 5 days of training
    history = train_model(network, X_train, Y_train_oh, batch_size, epochs)

    # Show progress (how well the dog is learning)
    print(history.history['loss'])
    print(history.history['accuracy'])
