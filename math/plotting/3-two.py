#!/usr/bin/env python3
"""
This module plots exponential decay graphs for two radioactive elements.
"""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plots two exponential decay graphs for C-14 and Ra-226.

    The graph:
    - Displays fraction remaining of C-14 and Ra-226 over time.
    - Includes:
        - X-axis labeled 'Time (years)'.
        - Y-axis labeled 'Fraction Remaining'.
        - Title: 'Exponential Decay of Radioactive Elements'.
        - Legend identifying C-14 and Ra-226.
    """
    x = np.arange(0, 21000, 1000)  # Time points
    r = np.log(0.5)  # Decay constant

    # Half-lives
    t_c14 = 5730  # C-14
    t_ra226 = 1600  # Ra-226

    # Fraction remaining calculations
    y_c14 = np.exp((r / t_c14) * x)
    y_ra226 = np.exp((r / t_ra226) * x)

    # Create the plot
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y_c14, 'r--', label="C-14")  # Dashed red line for C-14
    plt.plot(x, y_ra226, 'g-', label="Ra-226")  # Solid green line for Ra-226
    plt.xlim(0, 20000)  # X-axis range
    plt.ylim(0, 1)  # Y-axis range
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")
    plt.legend(loc="upper right")
    plt.show()
