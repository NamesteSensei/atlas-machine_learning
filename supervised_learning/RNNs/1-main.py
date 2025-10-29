#!/usr/bin/env python3
"""
Main test file for Task 1 - Forward propagation through a full RNN sequence.
"""

import numpy as np
RNNCell = __import__('0-rnn_cell').RNNCell
rnn = __import__('1-rnn').rnn

# Set random seed for reproducibility
np.random.seed(1)

# Instantiate an RNNCell with dimensions: input=10, hidden=15, output=5
rnn_cell = RNNCell(10, 15, 5)

# Randomize the cell’s biases for hidden and output layers
rnn_cell.bh = np.random.randn(1, 15)
rnn_cell.by = np.random.randn(1, 5)

# Create input data for 6 time steps, batch size 8, input size 10
X = np.random.randn(6, 8, 10)

# Initialize hidden state as zeros
h_0 = np.zeros((8, 15))

# Perform forward propagation
H, Y = rnn(rnn_cell, X, h_0)

# Display results
print(H.shape)
print(H)
print(Y.shape)
print(Y)
