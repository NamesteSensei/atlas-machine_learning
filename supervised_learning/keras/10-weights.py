#!/usr/bin/env python3
"""Module to save and load weights for a Keras model."""

import tensorflow.keras as K  # noqa: F401


def save_weights(network, filename, save_format='keras'):
    """
    Save the weights of a Keras model.

    Parameters
    ----------
    network : K.Model
        The model whose weights should be saved.
    filename : str
        The path of the file where the weights should be saved.
    save_format : str, optional
        Format to use when saving ('keras' or 'h5'), by default 'keras'.

    Returns
    -------
    None
    """
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """
    Load weights into a Keras model.

    Parameters
    ----------
    network : K.Model
        The model that should have its weights loaded.
    filename : str
        The path of the file containing the weights.

    Returns
    -------
    None
    """
    network.load_weights(filename)
