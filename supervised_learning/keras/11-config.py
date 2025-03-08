#!/usr/bin/env python3
"""
Module 11-config: Save and Load Model Configuration in JSON Format

This module provides functions to save a Keras model's configuration to a JSON
file and to load a model configuration from a JSON file.

Functions:
- save_config(network, filename): Saves a model’s configuration in JSON format.
- load_config(filename): Loads a model with a specific configuration from a file.
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format.

    Args:
        network (K.Model): The model to save the configuration of.
        filename (str): The path where the configuration should be saved.

    Returns:
        None
    """
    config = network.to_json()
    with open(filename, 'w') as file:
        file.write(config)


def load_config(filename):
    """
    Loads a model with a specific configuration from a JSON file.

    Args:
        filename (str): The path of the file with the model’s configuration.

    Returns:
        K.Model: The loaded model.
    """
    with open(filename, 'r') as file:
        config = file.read()
    model = K.models.model_from_json(config)
    return model
