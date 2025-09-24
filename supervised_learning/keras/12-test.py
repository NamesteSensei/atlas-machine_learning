#!/usr/bin/env python3
"""
12-test.py
Test a trained Keras model.
"""

def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network.

    Parameters
    ----------
    network : K.Model
        The network model to test.
    data : np.ndarray
        Input data to test the model with.
    labels : np.ndarray
        Correct one-hot labels of data.
    verbose : bool, optional
        Determines if output should be printed during testing.

    Returns
    -------
    list
        [loss, accuracy] of the model with the testing data.
    """
    return network.evaluate(data, labels, verbose=verbose)
