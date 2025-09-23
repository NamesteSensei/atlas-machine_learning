#!/usr/bin/env python3
"""
5-train.py
----------
Trains a Keras model using mini-batch gradient descent,
with optional validation data analysis.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent.

    Parameters
    ----------
    network : K.Model
        The Keras model to train.
    data : np.ndarray
        Input data of shape (m, nx).
    labels : np.ndarray
        One-hot labels of shape (m, classes).
    batch_size : int
        Size of each mini-batch.
    epochs : int
        Number of passes through the dataset.
    validation_data : tuple, optional
        Data to validate the model with (X_valid, Y_valid).
    verbose : bool, optional
        If True, print training progress. Default is True.
    shuffle : bool, optional
        If True, shuffle the dataset before each epoch.
        Default is False (for reproducibility).

    Returns
    -------
    History : keras.callbacks.History
        The History object generated after training.
    """
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle
    )
