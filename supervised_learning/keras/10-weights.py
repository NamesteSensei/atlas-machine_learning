#!/usr/bin/env python3

"""
Module for saving and loading model weights using Keras.
"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves a model's weights to a file.

    Args:
        network (K.Model): The model whose weights are to be saved.
        filename (str): The path of the file to save the weights to.
        save_format (str): The format to save the weights ('keras' or 'h5').
                           Defaults to 'h5'.
    Raises:
        ValueError: If save_format is not 'keras' or 'h5'.
    """
    if save_format not in ['keras', 'h5']:
        raise ValueError("Invalid save_format. Use 'keras' or 'h5'.")

    # Append the correct extension if needed:
    if save_format == 'keras':
        if not filename.endswith('.keras'):
            filename += '.keras'
    else:  # save_format == 'h5'
        if not filename.endswith('.weights.h5'):
            filename += '.weights.h5'

    # Save the weights (format is inferred from the file extension)
    network.save_weights(filename)
    print(f"Weights saved successfully to {filename}")


def load_weights(network, filename):
    """
    Loads a model's weights from a file.

    Args:
        network (K.Model): The model into which the weights will be loaded.
        filename (str): The path of the file from which weights are loaded.
    """
    # You might want to check for file existence here if desired.
    network.load_weights(filename)
    print(f"Weights loaded successfully from {filename}")
