#!/usr/bin/env python3
"""
Main file for testing 8-train.py with model checkpointing.
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


# Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model

if __name__ == '__main__':
    datasets = np.load('MNIST.npz')
    X_train = datasets['X_train'].reshape(datasets['X_train'].shape[0], -1)
    Y_train = one_hot(datasets['Y_train'])
    X_valid = datasets['X_valid'].reshape(datasets['X_valid'].shape[0], -1)
    Y_valid = one_hot(datasets['Y_valid'])

    network = build_model(784, [256, 256, 10],
                          ['relu', 'relu', 'softmax'],
                          lambtha=0.0001, keep_prob=0.95)

    optimize_model(network, alpha=0.001, beta1=0.9, beta2=0.999)

    train_model(network, X_train, Y_train, batch_size=64, epochs=1000,
                validation_data=(X_valid, Y_valid), early_stopping=True,
                patience=3, learning_rate_decay=True, alpha=0.001,
                save_best=True, filepath='network1.keras')
