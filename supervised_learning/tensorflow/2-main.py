#!/usr/bin/env python3
""" Test forward propagation """

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()  # Disable eager execution for TensorFlow v1 compatibility

# Import necessary functions
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop

# Create input placeholders (x: input, y: labels)
x, y = create_placeholders(784, 10)

# Define the structure of the neural network
layer_sizes = [256, 256, 10]  # Number of neurons in each layer
activations = [tf.nn.tanh, tf.nn.tanh, None]  # Activation functions

# Perform forward propagation
y_pred = forward_prop(x, layer_sizes, activations)

# Print the output tensor
print(y_pred)
