#!/usr/bin/env python3
import numpy as np

oh_encode = __import__('24-one_hot_encode').one_hot_encode

# Load a small sample from the MNIST dataset
lib = np.load('../data/MNIST.npz')
Y = lib['Y_train'][:10]

print("Original labels:")
print(Y)
Y_one_hot = oh_encode(Y, 10)
print("One-hot encoded:")
print(Y_one_hot)
