#!/usr/bin/env python3

"""
Module: 4-train
Trains a model using mini-batch gradient descent.
"""


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a Keras model using mini-batch gradient descent.

    Parameters:
    - network (keras.Model): The model to train.
    - data (numpy.ndarray): Input data of shape (m, nx).
    - labels (numpy.ndarray): One-hot encoded labels of shape (m, classes).
    - batch_size (int): Size of the batch for mini-batch gradient descent.
    - epochs (int): Number of passes through the data.
    - verbose (bool): If True, output is printed during training.
    - shuffle (bool): If True, shuffles batches every epoch.

    Returns:
    - History: The History object generated after training the model.
    """
    history = network.fit(
        data, labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
    )
    return history
