#!/usr/bin/env python3
"""
12-test.py
Test a trained neural network model
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a Keras model.

    Args:
        network: compiled Keras model to evaluate
        data: input data to test the model with
        labels: correct one-hot encoded labels of the data
        verbose: bool, if True, output will be printed during evaluation

    Returns:
        A list [loss, accuracy] of the model on the test data
    """
    return network.evaluate(data, labels, verbose=verbose)
