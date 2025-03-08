#!/usr/bin/env python3
"""
Main file for testing save and load configuration
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
model = __import__('9-model')
config = __import__('11-config')

if __name__ == '__main__':
    # Load the pre-trained model
    network = model.load_model('network1.keras')
    
    # Save the model configuration to a JSON file
    config.save_config(network, 'config1.json')
    del network  # Delete the model from memory
    
    # Load the model configuration from the JSON file
    network2 = config.load_config('config1.json')
    
    # Display the model summary
    network2.summary()
    
    # Print the weights of the loaded model
    print(network2.get_weights())
