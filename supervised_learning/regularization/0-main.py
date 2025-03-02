#!/usr/bin/env python3
import numpy as np
from l2_reg_cost import l2_reg_cost

if __name__ == '__main__':
    np.random.seed(0)
    
    # Initialize random weights for a 3-layer network
    weights = {
        'W1': np.random.randn(256, 784),
        'W2': np.random.randn(128, 256),
        'W3': np.random.randn(10, 128)
    }

    # Set a random initial cost
    cost = np.abs(np.random.randn(1))

    # Display original cost
    print("Original Cost:", cost)

    # Calculate and display the L2 regularized cost
    cost = l2_reg_cost(cost, 0.1, weights, 3, 1000)
    print("L2 Regularized Cost:", cost)
