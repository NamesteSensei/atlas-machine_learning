#!/usr/bin/env python3
"""
Test file for the Baum–Welch algorithm (Task 6)
"""

import numpy as np
baum_welch = __import__('6-baum_welch').baum_welch

if __name__ == "__main__":
    np.random.seed(1)

    # True emission and transition probabilities
    Emission = np.array([[0.90, 0.10, 0.00],
                         [0.40, 0.50, 0.10]])

    Transition = np.array([[0.60, 0.40],
                           [0.30, 0.70]])

    Initial = np.array([0.5, 0.5])

    # Generate hidden state sequence
    Hidden = [np.random.choice(2, p=Initial)]
    for _ in range(364):
        Hidden.append(np.random.choice(2, p=Transition[Hidden[-1]]))
    Hidden = np.array(Hidden)

    # Generate observation sequence
    Observations = [np.random.choice(3, p=Emission[s]) for s in Hidden]
    Observations = np.array(Observations)

    # Initialize test parameters
    T_test = np.ones((2, 2)) / 2
    E_test = np.abs(np.random.randn(2, 3))
    E_test = E_test / np.sum(E_test, axis=1).reshape((-1, 1))

    # Run Baum–Welch training
    T, E = baum_welch(Observations, T_test, E_test,
                      Initial.reshape((-1, 1)))

    # Display results
    print(np.round(T, 2))
    print(np.round(E, 2))
