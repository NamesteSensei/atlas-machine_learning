#!/usr/bin/env python3
"""
Module: 6-train
Trains a Keras model using mini-batch gradient descent with early stopping.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None,
                early_stopping=False, patience=0, verbose=True, shuffle=False):
    """
    Trains a Keras model using mini-batch gradient descent with optional 
    early stopping.

    Parameters:
    - network (keras.Model): The model to train.
    - data (numpy.ndarray): Input data of shape (m, nx).
    - labels (numpy.ndarray): One-hot encoded labels of shape (m, classes).
    - batch_size (int): Size of the batch for mini-batch gradient descent.
    - epochs (int): Number of passes through the data.
    - validation_data (tuple): Data to validate the model, if not None.
    - early_stopping (bool): Whether to use early stopping based on val_loss.
    - patience (int): The patience parameter for early stopping.
    - verbose (bool): If True, output is printed during training.
    - shuffle (bool): If True, shuffles batches every epoch.

    Returns:
    - History: The History object generated after training the model.
    """
    callbacks = []
    if early_stopping and validation_data:
        early_stopping_cb = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            mode='min'
        )
        callbacks.append(early_stopping_cb)

    history = network.fit(
        data, labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )
    return history
