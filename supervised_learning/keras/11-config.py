#!/usr/bin/env python3
"""
11-config.py
Save and load a model configuration in JSON format
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format.

    Args:
        network: Keras model to save the config from.
        filename: Path to file where the JSON config should be saved.

    Returns:
        None
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)


def load_config(filename):
    """
    Loads a model from a saved JSON configuration.

    Args:
        filename: Path to JSON config file.

    Returns:
        A Keras model built from the config (uncompiled, with no weights).
    """
    with open(filename, 'r') as f:
        config = f.read()
    return K.models.model_from_json(config)
