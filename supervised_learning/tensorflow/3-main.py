#!/usr/bin/env python3
""" Test accuracy calculation """

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# Import necessary functions
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy

# Create placeholders (input x, labels y)
x, y = create_placeholders(784, 10)

# Define the neural network
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])

# Calculate accuracy
accuracy = calculate_accuracy(y, y_pred)

# Print the accuracy tensor
print(accuracy)
