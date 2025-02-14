#!/usr/bin/env python3
""" Test for the training operation """

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


# Create placeholders for input data
x, y = create_placeholders(784, 10)

# Build the forward propagation model
y_pred = forward_prop(x, [256, 256, 10], [tf.nn.tanh, tf.nn.tanh, None])

# Calculate the loss
loss = calculate_loss(y, y_pred)

# Create the training operation
train_op = create_train_op(loss, 0.01)

# Print the training operation to verify
print(train_op)
