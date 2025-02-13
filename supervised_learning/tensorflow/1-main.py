#!/usr/bin/env python3
"""Test script for Task 1 - create_layer function"""

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()

# Import functions
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_layer = __import__('1-create_layer').create_layer

# Create placeholders
x, y = create_placeholders(784, 10)

# Create a layer
l = create_layer(x, 256, tf.nn.tanh)

# Print layer output
print()
