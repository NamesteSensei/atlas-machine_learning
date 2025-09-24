#!/usr/bin/env python3
"""
13-predict.py
Make a prediction using a trained Keras model
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Uses a Keras model to make predictions.

    Args:
        network: trained Keras model
        data: input data to predict
        verbose: bool, whether to print output during prediction

    Returns:
        The model’s prediction on the data
    """
    return network.predict(data, verbose=verbose)
