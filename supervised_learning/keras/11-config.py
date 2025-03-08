#!/usr/bin/env python3
import tensorflow.keras as K

def save_config(network, filename):
    """
    Saves a model’s configuration in JSON format.
    
    Args:
        network (K.Model): The model whose configuration should be saved.
        filename (str): The path of the file to save the configuration.
    
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
        filename (str): The path of the file containing the model’s configuration.
    
    Returns:
        K.Model: The loaded model.
    """
    with open(filename, 'r') as f:
        config = f.read()
    model = K.models.model_from_json(config)
    return model
