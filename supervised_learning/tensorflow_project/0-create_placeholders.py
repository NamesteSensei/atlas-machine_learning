#!/usr/bin/env python3
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for a neural network.

    Args:
    - nx: Number of feature columns (input size)
    - classes: Number of output classes (labels)

    Returns:
    - x: Placeholder for input data
    - y: Placeholder for one-hot labels
    """
    x = tf.placeholder(dtype=tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(dtype=tf.float32, shape=(None, classes), name="y")
    return x, y

