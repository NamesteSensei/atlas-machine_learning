#!/usr/bin/env python3
"""
Module for making predictions with a Keras neural network model.

Provides a function to predict using a trained model with optional verbose
output.
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network model.

    Args:
        network (K.Model): The trained model to make predictions with.
        data (np.ndarray): The input data to make predictions on.
        verbose (bool, optional): If True, print prediction process output.
                                  Defaults to False.

    Returns:
        np.ndarray: The predicted output for the input data.
    """
    return network.predict(data, verbose=verbose)
