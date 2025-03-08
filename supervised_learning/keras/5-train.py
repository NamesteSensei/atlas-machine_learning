#!/usr/bin/env python3

"""
Module: 5-train
Trains a model using mini-batch gradient descent with validation.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a Keras model using mini-batch gradient descent with validation.

    Parameters:
    - network (keras.Model): The model to train.
    - data (numpy.ndarray): Input data of shape (m, nx).
    - labels (numpy.ndarray): One-hot encoded labels (m, classes).
    - batch_size (int): Size of the batch for mini-batch gradient descent.
    - epochs (int): Number of passes through the data.
    - validation_data (tuple): Optional validation data (X_val, Y_val).
    - verbose (bool): If True, output is printed during training.
    - shuffle (bool): If True, shuffles batches every epoch.

    Returns:
    - History: The History object from model training.
    """
    history = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data
    )
    return history
