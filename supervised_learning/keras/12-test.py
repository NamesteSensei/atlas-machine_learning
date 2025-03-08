#!/usr/bin/env python3
"""
Module 12-test: Test a Neural Network Model

This module provides a function to test a Keras model using provided test data 
and labels. It returns the loss and accuracy of the model on the test data.

Functions:
- test_model(network, data, labels, verbose=True): Tests a Keras model.
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network model.

    Args:
        network (K.Model): The network model to test.
        data (numpy.ndarray): The input data for testing the model.
        labels (numpy.ndarray): The correct one-hot labels for the data.
        verbose (bool): If True, prints output during testing.

    Returns:
        list: The loss and accuracy of the model on the test data.
    """
    return network.evaluate(data, labels, verbose=verbose)
