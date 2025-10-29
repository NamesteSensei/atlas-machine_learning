#!/usr/bin/env python3
"""
Main test file for Task 2 - GRUCell
Tests initialization and forward propagation of a GRU cell.
"""

import numpy as np
GRUCell = __import__('2-gru_cell').GRUCell

# Set seed for reproducibility
np.random.seed(2)

# Create GRU cell with input=10, hidden=15, output=5
gru_cell = GRUCell(10, 15, 5)

# Print initialized weights and biases
print("Wz:", gru_cell.Wz)
print("Wr:", gru_cell.Wr)
print("Wh:", gru_cell.Wh)
print("Wy:", gru_cell.Wy)
print("bz:", gru_cell.bz)
print("br:", gru_cell.br)
print("bh:", gru_cell.bh)
print("by:", gru_cell.by)

# Randomize the biases (to match project checker values)
gru_cell.bz = np.random.randn(1, 15)
gru_cell.br = np.random.randn(1, 15)
gru_cell.bh = np.random.randn(1, 15)
gru_cell.by = np.random.randn(1, 5)

# Create random inputs for testing
h_prev = np.random.randn(8, 15)  # previous hidden state
x_t = np.random.randn(8, 10)     # current input data

# Perform one forward step
h, y = gru_cell.forward(h_prev, x_t)

# Print results
print(h.shape)
print(h)
print(y.shape)
print(y)
