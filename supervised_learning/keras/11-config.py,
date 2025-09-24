#!/usr/bin/env python3
"""
11-config.py
Save and load a model’s configuration (architecture only).
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format.

    Parameters
    ----------
    network : K.Model
        The model to save.
    filename : str
        Path where the configuration should be saved.

    Returns
    -------
    None
    """
    config = network.to_json()
    with open(filename, "w") as f:
        f.write(config)


def load_config(filename):
    """
    Loads a model’s configuration from a JSON file.

    Parameters
    ----------
    filename : str
        Path to the saved JSON configuration file.

    Returns
    -------
    K.Model
        The loaded Keras model architecture.
    """
    with open(filename, "r") as f:
        config = f.read()
    return K.models.model_from_json(config)
