#!/usr/bin/env python3

"""Sets up Adam optimization for a Keras model."""

from keras.optimizers import Adam


def optimize_model(network, alpha, beta1, beta2):
    """
    Configures the model for training using the Adam optimizer
    with categorical crossentropy loss and accuracy metrics.

    Args:
        network: The Keras model to optimize.
        alpha: The learning rate.
        beta1: The first Adam optimization parameter.
        beta2: The second Adam optimization parameter.

    Returns:
        None
    """
    optimizer = Adam(learning_rate=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
