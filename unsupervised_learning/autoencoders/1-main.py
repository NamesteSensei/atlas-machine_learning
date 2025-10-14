#!/usr/bin/env python3
"""
Tests the sparse autoencoder on MNIST dataset
"""

import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

autoencoder = __import__('1-sparse').autoencoder

# === Seed for reproducibility
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# === Load and preprocess MNIST data
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

# === Build the sparse autoencoder
encoder, decoder, auto = autoencoder(
    input_dims=784,
    hidden_layers=[128, 64],
    latent_dims=32,
    lambtha=10e-6
)

# === Train the model
auto.fit(
    x_train, x_train,
    epochs=50,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# === Encode and decode a few samples
encoded = encoder.predict(x_test[:10])
print(np.mean(encoded))
reconstructed = decoder.predict(encoded)

# === Plot original and reconstructed images
for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i].reshape((28, 28)))
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(reconstructed[i].reshape((28, 28)))
plt.show()
