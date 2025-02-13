#!/usr/bin/env python3
"""
0-create_placeholders.py
This script defines a function to create TensorFlow placeholders for input data and labels.
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # Ensures TensorFlow v1 compatibility

def create_placeholders(nx, classes):
    """
    Creates placeholders for the neural network.

    Parameters:
    nx (int): Number of feature columns in the input data.
    classes (int): Number of classes in the classifier.

    Returns:
    x (tf.placeholder): Placeholder for input data (shape: [None, nx])
    y (tf.placeholder): Placeholder for one-hot labels (shape: [None, classes])
    """
    x = tf.placeholder(dtype=tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(dtype=tf.float32, shape=(None, classes), name="y")
    return x, y

# Only runs when the script is executed directly (for testing)
if __name__ == "__main__":
    x, y = create_placeholders(784, 10)
    print(x)
    print(y)
