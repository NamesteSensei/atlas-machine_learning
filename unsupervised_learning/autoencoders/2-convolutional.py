#!/usr/bin/env python3
"""Vanilla convolutional autoencoder"""

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder

    Args:
        input_dims (tuple): shape of the input (height, width, channels)
        filters (list): number of filters for encoder layers
        latent_dims (tuple): shape of the latent space representation

    Returns:
        encoder (Model): encoder model
        decoder (Model): decoder model
        auto (Model): full autoencoder model
    """
    input_img = keras.Input(shape=input_dims)

    x = input_img
    for f in filters:
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                padding='same', activation='relu')(x)
        x = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                      padding='same')(x)

    encoder = keras.Model(inputs=input_img, outputs=x)

    latent_input = keras.Input(shape=latent_dims)
    x = latent_input

    for f in reversed(filters[:-1]):
        x = keras.layers.Conv2D(filters=f, kernel_size=(3, 3),
                                padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    # Penultimate conv layer: same filters as first encoder layer
    x = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                            padding='valid', activation='relu')(x)

    # Final conv layer to match input channels
    x = keras.layers.UpSampling2D(size=(2, 2))(x)
    x = keras.layers.Conv2D(filters=input_dims[2], kernel_size=(3, 3),
                            padding='same', activation='sigmoid')(x)

    decoder = keras.Model(inputs=latent_input, outputs=x)

    auto_input = keras.Input(shape=input_dims)
    encoded = encoder(auto_input)
    decoded = decoder(encoded)

    auto = keras.Model(inputs=auto_input, outputs=decoded)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
