#!/usr/bin/env python3
"""Convolutional autoencoder implementation"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Builds a convolutional autoencoder.

    Args:
        input_dims (tuple): shape of the input image (h, w, c)
        filters (list): filters for encoder conv layers
        latent_dims (tuple): latent space shape

    Returns:
        encoder (Model): encoder model
        decoder (Model): decoder model
        auto (Model): full autoencoder
    """
    # === Encoder ===
    input_img = keras.Input(shape=input_dims)
    x = input_img
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(inputs=input_img, outputs=x)

    # === Decoder ===
    latent_input = keras.Input(shape=latent_dims)
    x = latent_input
    # reverse filters for symmetric structure
    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(f, (3, 3), activation='relu',
                                padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    # last conv before final output (valid padding)
    x = keras.layers.Conv2D(filters[-1], (3, 3), activation='relu',
                            padding='valid')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    x = keras.layers.Conv2D(input_dims[-1], (3, 3),
                            activation='sigmoid', padding='same')(x)
    decoder = keras.Model(inputs=latent_input, outputs=x)

    # === Autoencoder ===
    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)
    auto = keras.Model(inputs=auto_input, outputs=decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
