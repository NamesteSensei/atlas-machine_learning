#!/usr/bin/env python3
""" Main script to test forward propagation """

import numpy as np
Neuron = __import__('2-neuron').Neuron

np.random.seed(0)
nx = 5
neuron = Neuron(nx)

X = np.random.randn(nx, 3)  # Example input with 3 samples
A = neuron.forward_prop(X)

print("Activated Output (A):")
print(A)
