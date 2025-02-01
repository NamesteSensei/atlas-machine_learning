#!/usr/bin/env python3
""" Main script to test the Neuron class """

import numpy as np
Neuron = __import__('0-neuron').Neuron

# Simulate dataset
np.random.seed(0)
nx = 5  # Number of input features
neuron = Neuron(nx)

# Print initialized values
print("Weights:\n", neuron.W)
print("Shape of Weights:", neuron.W.shape)
print("Bias:", neuron.b)
print("Activated Output:", neuron.A)

# Modify A and check
neuron.A = 10
print("Modified A:", neuron.A)
