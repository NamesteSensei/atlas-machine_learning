#!/usr/bin/env python3
"""
Script to test the Neuron class with multiple epochs.
"""

import numpy as np
import matplotlib.pyplot as plt
Neuron = __import__('6-neuron').Neuron

# Load the dataset
lib_train = np.load('../data/train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T  # Reshape input data

# Initialize neuron
np.random.seed(0)
neuron = Neuron(X.shape[0])

# Train the neuron with multiple epochs
A, cost_history = neuron.train(X, Y, iterations=500, alpha=0.05)

# Display results
print("\nFinal Prediction:")
print(A)

# Ensure cost_history is a list before accessing [-1]
if isinstance(cost_history, list) and len(cost_history) > 0:
    print("\nFinal Cost:", cost_history[-1])
else:
    print("\nFinal Cost: No cost history recorded!")

# Plot cost history if available
if isinstance(cost_history, list) and len(cost_history) > 1:
    plt.plot(range(0, len(cost_history) * 100, 100), cost_history, marker='o', linestyle='-')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Training Cost over Iterations')
    plt.show()
