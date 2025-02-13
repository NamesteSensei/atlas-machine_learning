#!/usr/bin/env python3
"""Module for creating TensorFlow placeholders"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()


def create_placeholders(nx, classes):
    """
    Function that returns two placeholders,
    x and y, for the neural network

    Args:
        nx (int): Number of feature columns in data
        classes (int): Number of classes in classifier

    Returns:
        tuple: x, y placeholders
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name='x')
    y = tf.placeholder(tf.float32, shape=(None, classes), name='y')

    return x, y


# Testing placeholders
if __name__ == "__main__":
    x, y = create_placeholders(784, 10)
    print(x)
    print(y)
