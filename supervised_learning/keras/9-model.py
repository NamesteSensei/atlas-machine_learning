#!/usr/bin/env python3
"""
Module: 9-model
Provides functions to save and load a Keras model.
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves an entire Keras model.

    Parameters:
    - network (keras.Model): The model to save.
    - filename (str): The path where the model should be saved.

    Returns:
    - None
    """
    K.models.save_model(network, filename)


def load_model(filename):
    """
    Loads an entire Keras model.

    Parameters:
    - filename (str): The path to load the model from.

    Returns:
    - keras.Model: The loaded model.
    """
    return K.models.load_model(filename)
