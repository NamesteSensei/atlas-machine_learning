#!/usr/bin/env python3
"""
Main file
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

# Imports from your project modules
one_hot = __import__('3-one_hot').one_hot
train_model = __import__('8-train').train_model
model = __import__('9-model')
weights = __import__('10-weights')

if __name__ == '__main__':

    # Load a trained model from file (network2.keras)
    network = model.load_model('network2.keras')
    
    # Save the model's weights using the HDF5 format.
    # This will save the weights to a file named 'weights2.weights.h5'
    weights.save_weights(network, 'weights2', save_format='h5')
    del network

    # Load a new instance of the model (from network1.keras)
    network2 = model.load_model('network1.keras')
    
    # Print the weights before loading (they will differ from network2's weights)
    print(network2.get_weights())

    # Now load the saved weights into network2.
    weights.load_weights(network2, 'weights2.weights.h5')
    
    # Print the weights after loading; they should now match those from network2.keras.
    print(network2.get_weights())
