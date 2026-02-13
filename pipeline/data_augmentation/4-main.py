#!/usr/bin/env python3

import matplotlib
matplotlib.use('TkAgg')

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

change_brightness = __import__('4-brightness').change_brightness

tf.random.set_seed(4)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)

for image, _ in doggies.shuffle(10).take(1):
    adjusted = change_brightness(image, 0.3)

    plt.imshow(adjusted)
    plt.axis('off')
    plt.title("Random Brightness Adjustment")
    plt.show()

print("Done.")
