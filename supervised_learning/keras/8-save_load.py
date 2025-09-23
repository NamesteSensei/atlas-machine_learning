#!/usr/bin/env python3
"""
8-save_load.py
--------------
Functions to save and load a Keras model.
"""

import tensorflow.keras as K
from datetime import datetime


def save_model(network, filename=None):
    """
    Saves an entire Keras model to an HDF5 file.

    Parameters
    ----------
    network : K.Model
        The model to save.
    filename : str, optional
        The path of the file to save the model.
        If None, generates a timestamped filename.

    Returns
    -------
    str
        The filename used for saving.
    """
    if filename is None:
        filename = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"

    network.save(filename)
    return filename


def load_model(filename):
    """
    Loads a Keras model from file.

    Parameters
    ----------
    filename : str
        The path of the saved model file.

    Returns
    -------
    K.Model
        The loaded Keras model.
    """
    return K.models.load_model(filename)
