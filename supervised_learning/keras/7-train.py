#!/usr/bin/env python3
"""
Module: 7-train
Trains a model using mini-batch gradient descent with optional early stopping
and learning rate decay.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Trains a Keras model using mini-batch gradient descent with optional
    early stopping and learning rate decay.

    Parameters:
    - network (keras.Model): The model to train.
    - data (numpy.ndarray): Input data of shape (m, nx).
    - labels (numpy.ndarray): One-hot encoded labels of shape (m, classes).
    - batch_size (int): Size of the batch for mini-batch gradient descent.
    - epochs (int): Number of passes through the data.
    - validation_data (tuple): Data to validate the model with, if not None.
    - early_stopping (bool): Indicates if early stopping should be used.
    - patience (int): Patience for early stopping.
    - learning_rate_decay (bool): If True, applies learning rate decay.
    - alpha (float): Initial learning rate.
    - decay_rate (float): Decay rate for learning rate.
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
            mode='min',
            verbose=verbose
        )
        callbacks.append(early_stopping_cb)

    if learning_rate_decay and validation_data:
        def lr_scheduler(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_decay_cb = K.callbacks.LearningRateScheduler(
            lr_scheduler,
            verbose=1
        )
        callbacks.append(lr_decay_cb)

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
