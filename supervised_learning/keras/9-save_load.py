#!/usr/bin/env python3
"""
9-save_load.py
--------------
Functions to save and load the weights of a Keras model.
"""

import tensorflow.keras as K


def save_weights(network, filename):
    """
    Saves the weights of a model.

    Parameters
    ----------
    network : K.Model
        The model whose weights should be saved.
    filename : str
        Path to the file where the weights should be saved.

    Returns
    -------
    None
    """
    network.save_weights(filename)


def load_weights(network, filename):
    """
    Loads weights into a model.

    Parameters
    ----------
    network : K.Model
        The model to which the weights should be loaded.
    filename : str
        Path of the file containing the saved weights.

    Returns
    -------
    None
    """
    network.load_weights(filename)
