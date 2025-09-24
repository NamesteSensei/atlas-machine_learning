#!/usr/bin/env python3
"""Utilities to save and load a full Keras model (architecture + weights)."""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Save an entire Keras model.

    Parameters
    ----------
    network : K.Model
        The model to save.
    filename : str
        Path where the model should be saved (.keras or .h5).

    Returns
    -------
    None
    """
    network.save(filename)


def load_model(filename):
    """
    Load a full Keras model from file.

    Parameters
    ----------
    filename : str
        Path to the saved model file (.keras or .h5).

    Returns
    -------
    K.Model
        The loaded Keras model.
    """
    return K.models.load_model(filename)
