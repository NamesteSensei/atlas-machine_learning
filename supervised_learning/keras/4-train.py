#!/usr/bin/env python3

"""
Module: 4-train
Trains a model using mini-batch gradient descent.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a Keras model using mini-batch gradient descent.

    Parameters:
    - network (K.Model): The model to train.
    - data (np.ndarray): Input data of shape (m, nx).
    - labels (np.ndarray): One-hot encoded labels of shape (m, classes).
    - batch_size (int): Size of the batch for mini-batch gradient descent.
    - epochs (int): Number of passes through the data.
    - verbose (bool): If True, output is printed during training.
    - shuffle (bool): If True, shuffles batches every epoch.

    Returns:
    - K.callbacks.History: The History object generated after training.
    """
    history = network.fit(data, labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
