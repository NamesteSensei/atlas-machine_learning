#!/usr/bin/env python3
"""
Creates a sparse autoencoder using L1 activity regularization
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Builds a sparse autoencoder

    Args:
        input_dims (int): size of input layer
        hidden_layers (list): encoder hidden layer sizes
        latent_dims (int): size of latent space
        lambtha (float): L1 regularization parameter

    Returns:
        encoder (Model): encoder model
        decoder (Model): decoder model
        auto (Model): full autoencoder
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(
        latent_dims, activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(x)
    encoder = keras.Model(inputs=inputs, outputs=latent)

    # Decoder
    latent_inputs = keras.Input(shape=(latent_dims,))
    x = latent_inputs
    for units in reversed(hidden_layers):
        x = keras.layers.Dense(units, activation='relu')(x)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(x)
    decoder = keras.Model(inputs=latent_inputs, outputs=outputs)

    # Autoencoder
    auto_inputs = keras.Input(shape=(input_dims,))
    encoded = encoder(auto_inputs)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=auto_inputs, outputs=decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')
    return encoder, decoder, auto
