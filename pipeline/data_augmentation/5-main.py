#!/usr/bin/env python3

import matplotlib
matplotlib.use('TkAgg')

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

change_hue = __import__('5-hue').change_hue

tf.random.set_seed(5)

doggies = tfds.load('stanford_dogs', split='train', as_supervised=True)

for image, _ in doggies.shuffle(10).take(1):
    adjusted = change_hue(image, -0.5)

    plt.imshow(adjusted)
    plt.axis('off')
    plt.title("Hue Adjusted Image")
    plt.show()

print("Done.")
