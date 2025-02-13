#!/usr/bin/env python3
"""
/** 
 * 0-main.py:
 * This file tests the creation of placeholders for the input data and one-hot labels.
 */
"""

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Import the create_placeholders function from module 0
create_placeholders = __import__('0-create_placeholders').create_placeholders

# Create placeholders with 784 features and 10 classes
x, y = create_placeholders(784, 10)
print(x)
print(y)
