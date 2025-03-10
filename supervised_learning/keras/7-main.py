#!/usr/bin/env python3
"""
Main file for Task 7: Model Training with Early Stopping and Learning Rate de
"""

import os
import random
import numpy as np
import tensorflow as tf

# Force Seed - fix for Keras
SEED = 8
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Imports from local modules
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('7-train').train_model

if __name__ == '__main__':
    datasets = np.load('MNIST.npz')
    X_train = datasets['X_train'].reshape(datasets['X_train'].shape[0], -1)
    Y_train = datasets['Y_train']
    Y_train_oh = one_hot(Y_train)

    X_valid = datasets['X_valid'].reshape(datasets['X_valid'].shape[0], -1)
    Y_valid = datasets['Y_valid']
    Y_valid_oh = one_hot(Y_valid)

    lambtha = 0.0001
    keep_prob = 0.95
    network = build_model(
        784, [256, 256, 10],
        ['relu', 'relu', 'softmax'],
        lambtha, keep_prob
    )

    alpha = 0.001
    beta1 = 0.9
    beta2 = 0.999
    optimize_model(network, alpha, beta1, beta2)

    batch_size = 64
    epochs = 1000

    train_model(
        network, X_train, Y_train_oh, batch_size, epochs,
        validation_data=(X_valid, Y_valid_oh),
        early_stopping=True, patience=3,
        learning_rate_decay=True, alpha=alpha, decay_rate=1
    )
