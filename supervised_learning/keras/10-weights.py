#!/usr/bin/env python3
"""
Module for saving and loading model weights using Keras.
"""

import tensorflow.keras as K

def save_weights(network, filename, save_format='h5'):
    """
    Saves a model's weights to a file.
    
    Args:
        network (K.Model): The model whose weights should be saved.
        filename (str): The base filename (or full filename) where weights will be saved.
        save_format (str): The format to save the weights ('keras' or 'h5').
                           Defaults to 'h5'.
    
    Raises:
        ValueError: If save_format is not 'keras' or 'h5'.
    """
    if save_format not in ['keras', 'h5']:
        raise ValueError("Invalid save_format. Use 'keras' or 'h5'.")

    # Append the proper extension if not already present.
    # For HDF5 format we expect the file to end with '.weights.h5'
    if save_format == 'keras':
        if not filename.endswith('.keras'):
            filename += '.keras'
    else:  # save_format == 'h5'
        if not filename.endswith('.weights.h5'):
            filename += '.weights.h5'
    
    network.save_weights(filename)
    print(f"Weights saved successfully to {filename}")

def load_weights(network, filename):
    """
    Loads a model's weights from a file.
    
    Args:
        network (K.Model): The model into which the weights will be loaded.
        filename (str): The path of the file from which to load weights.
    """
    network.load_weights(filename)
    print(f"Weights loaded successfully from {filename}")
