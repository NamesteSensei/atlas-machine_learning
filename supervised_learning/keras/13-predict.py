#!/usr/bin/env python3
"""
13-predict.py
Make predictions with a trained Keras model.
"""

def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network.

    Parameters
    ----------
    network : K.Model
        The network model to make the prediction with.
    data : np.ndarray
        Input data to make the prediction with.
    verbose : bool, optional
        Determines if output should be printed during prediction.

    Returns
    -------
    np.ndarray
        The prediction for the data.
    """
    return network.predict(data, verbose=verbose)
