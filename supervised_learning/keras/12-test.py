#!/usr/bin/env python3
"""
Module for testing a Keras neural network model.

Provides a function to evaluate a neural network model using test data
and return the loss and accuracy of the model.
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network model.

    Args:
        network (K.Model): The model to test.
        data (np.ndarray): The input data for testing.
        labels (np.ndarray): The correct one-hot labels for the data.
        verbose (bool, optional): If True, print testing output. 
                                  Defaults to True.

    Returns:
        list: The loss and accuracy of the model on the test data.
    """
    return network.evaluate(data, labels, verbose=verbose)
