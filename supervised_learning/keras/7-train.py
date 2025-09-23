#!/usr/bin/env python3
"""
7-train.py
----------
Trains a Keras model using mini-batch gradient descent,
with optional validation, early stopping, and learning rate decay.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.001, decay_rate=1, verbose=True, shuffle=False):
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
    early_stopping : bool, optional
        If True and validation_data is provided, enable early stopping.
    patience : int, optional
        Number of epochs to wait after no improvement before stopping.
    learning_rate_decay : bool, optional
        If True and validation_data is provided, enable learning rate decay.
    alpha : float, optional
        Initial learning rate.
    decay_rate : float, optional
        Decay rate for inverse time decay schedule.
    verbose : bool, optional
        If True, print training progress. Default is True.
    shuffle : bool, optional
        If True, shuffle the dataset before each epoch.

    Returns
    -------
    History : keras.callbacks.History
        The History object generated after training.
    """
    callbacks = []

    # Early Stopping
    if early_stopping and validation_data is not None:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stop)

    # Learning Rate Decay
    if learning_rate_decay and validation_data is not None:
        def scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_decay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callbacks.append(lr_decay)

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )
