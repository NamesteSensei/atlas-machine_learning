#!/usr/bin/env python3
"""
Creates a vanilla autoencoder with fully connected layers
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Builds a fully connected autoencoder

    Args:
        input_dims (int): size of input layer
        hidden_layers (list): nodes in encoder hidden layers
        latent_dims (int): size of latent space

    Returns:
        encoder (Model): encoder model
        decoder (Model): decoder model
        auto (Model): full autoencoder model
    """
    # Encoder
    inputs = keras.Input(shape=(input_dims,))
    x = inputs
    for units in hidden_layers:
        x = keras.layers.Dense(units, activation='relu')(x)
    latent = keras.layers.Dense(latent_dims, activation='relu')(x)
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
