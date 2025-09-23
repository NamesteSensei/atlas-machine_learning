#!/usr/bin/env python3
"""
2-optimize.py
-------------
Sets up Adam optimization for a Keras model
with categorical crossentropy loss and accuracy metric.
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Configures a Keras model for training using Adam optimization.

    Parameters
    ----------
    network : K.Model
        The Keras model to compile.
    alpha : float
        Learning rate for the Adam optimizer.
    beta1 : float
        First Adam optimization parameter.
    beta2 : float
        Second Adam optimization parameter.

    Returns
    -------
    None
    """
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )

    network.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
