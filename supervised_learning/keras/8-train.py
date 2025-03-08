#!/usr/bin/env python3
"""
Module: 8-train
Trains a model using mini-batch gradient descent with optional
early stopping, learning rate decay, and model checkpointing.
"""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1,
                save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a Keras model with optional early stopping, learning rate decay,
    and model checkpointing.

    Parameters:
    - network (keras.Model): The model to train.
    - data (numpy.ndarray): Input data of shape (m, nx).
    - labels (numpy.ndarray): One-hot encoded labels of shape (m, classes).
    - batch_size (int): Size of the batch for mini-batch gradient descent.
    - epochs (int): Number of passes through the data.
    - validation_data (tuple): Data to validate the model.
    - early_stopping (bool): Whether to apply early stopping.
    - patience (int): Number of epochs with no improvement before stopping.
    - learning_rate_decay (bool): Whether to apply learning rate decay.
    - alpha (float): Initial learning rate.
    - decay_rate (float): Decay rate for learning rate decay.
    - save_best (bool): Whether to save the best model based on validation.
    - filepath (str): File path to save the best model.
    - verbose (bool): Whether to print output during training.
    - shuffle (bool): Whether to shuffle batches every epoch.

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

    if save_best and filepath:
        checkpoint_cb = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=verbose
        )
        callbacks.append(checkpoint_cb)

    history = network.fit(
        data, labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
    return history
