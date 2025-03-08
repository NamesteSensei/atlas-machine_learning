#!/usr/bin/env python3
"""
Module for saving and loading model weights in TensorFlow Keras.
"""

import tensorflow as tf

def save_weights(network, filename, save_format='tf'):
    """
    Saves the model's weights to the specified file.

    Args:
        network (tf.keras.Model): The model whose weights are to be saved.
        filename (str): The path to the file where weights will be saved.
        save_format (str): The format to save the weights ('tf' or 'h5').
                           Defaults to 'tf'.
    """
    # Validate save_format
    if save_format not in ['tf', 'h5']:
        raise ValueError("Invalid save_format. Use 'tf' or 'h5'.")

    # Ensure filename matches expected format
    if save_format == 'h5' and not filename.endswith('.h5'):
        filename += '.h5'
    elif save_format == 'tf' and not filename.endswith('.keras'):
        filename += '.keras'

    # Save weights using the correct format
    network.save_weights(filename)


def load_weights(network, filename):
    """
    Loads the model's weights from the specified file.

    Args:
        network (tf.keras.Model): The model to which the weights will be loaded.
        filename (str): The path to the file from which weights are loaded.
    """
    network.load_weights(filename)
