#!/usr/bin/env python3
""" Main script to test the Neuron class """

import numpy as np
Neuron = __import__('1-neuron').Neuron

np.random.seed(0)
nx = 5
neuron = Neuron(nx)

print("Weights:\n", neuron.W)
print("Bias:", neuron.b)
print("Activated Output:", neuron.A)

# Attempt to modify private attributes
try:
    neuron.W = 10
except AttributeError as e:
    print("[Error]:", e)

try:
    neuron.b = 10
except AttributeError as e:
    print("[Error]:", e)

try:
    neuron.A = 10
except AttributeError as e:
    print("[Error]:", e)
