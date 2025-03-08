#!/usr/bin/env python3

import json
from tensorflow import keras

def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format.

    Args:
        network (keras.Model): The model whose configuration should be saved.
        filename (str): The path of the file that the configuration should be saved to.

    Returns:
        None
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)

def load_config(filename):
    """
    Loads a model with a specific configuration from a JSON file.

    Args:
        filename (str): The path of the file containing the model’s configuration in JSON format.

    Returns:
        keras.Model: The loaded model.
    """
    with open(filename, 'r') as f:
        config = f.read()
    model = keras.models.model_from_json(config)
    return model
