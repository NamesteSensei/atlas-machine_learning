#!/usr/bin/env python3
"""
Main file for testing 6-train module
"""

# Imports
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

# Module Imports
build_model = __import__('1-input').build_model
optimize_model = __import__('2-optimize').optimize_model
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('6-train').train_model

if __name__ == '__main__':
    datasets = np.load('MNIST.npz')
    X_train = datasets['X_train'].reshape(-1, 784)
    Y_train = one_hot(datasets['Y_train'])
    X_valid = datasets['X_valid'].reshape(-1, 784)
    Y_valid = one_hot(datasets['Y_valid'])

    network = build_model(
        784, [256, 256, 10], ['relu', 'relu', 'softmax'],
        lambtha=0.0001, keep_prob=0.95
    )
    optimize_model(network, alpha=0.001, beta1=0.9, beta2=0.999)

    train_model(
        network, X_train, Y_train,
        batch_size=64, epochs=30,
        validation_data=(X_valid, Y_valid),
        early_stopping=True, patience=3
    )
