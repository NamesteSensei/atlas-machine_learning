#!/usr/bin/env python3
"""
Test script for vanilla autoencoder using MNIST
"""
import os
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import random
random.seed(0)
import numpy as np
np.random.seed(0)
import tensorflow as tf
tf.random.set_seed(0)

import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

autoencoder = __import__('0-vanilla').autoencoder

(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((-1, 784))
x_test = x_test.reshape((-1, 784))

enc, dec, auto = autoencoder(784, [128, 64], 32)

auto.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True,
         validation_data=(x_test, x_test), verbose=0)

encoded = enc.predict(x_test[:10])
print(np.mean(encoded))
recon = dec.predict(encoded)

for i in range(10):
    ax = plt.subplot(2, 10, i + 1)
    ax.axis('off')
    plt.imshow(x_test[i].reshape((28, 28)))
    ax = plt.subplot(2, 10, i + 11)
    ax.axis('off')
    plt.imshow(recon[i].reshape((28, 28)))
plt.show()
