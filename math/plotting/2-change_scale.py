#!/usr/bin/env python3
"""
This module plots an exponential decay graph of C-14.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots an exponential decay graph of C-14.

    The graph:
    - Displays the fraction remaining of C-14 over time.
    - Uses a logarithmic scale on the y-axis.
    - Includes:
        - X-axis labeled 'Time (years)'.
        - Y-axis labeled 'Fraction Remaining'.
        - Title: 'Exponential Decay of C-14'.
    """
    # Generate data
    x = np.arange(0, 28651, 5730)  # Time points in years
    r = np.log(0.5)  # Decay constant
    t = 5730  # Half-life of C-14
    y = np.exp((r / t) * x)  # Fraction remaining

    # Create line graph
    plt.figure(figsize=(6.4, 4.8))  # Set figure size
    plt.plot(x, y, 'r-')  # Plot a solid red line
    plt.xlim(0, 28650)  # Set x-axis range
    plt.yscale('log')  # Set y-axis to logarithmic scale
    plt.xlabel("Time (years)")  # Label x-axis
    plt.ylabel("Fraction Remaining")  # Label y-axis
    plt.title("Exponential Decay of C-14")  # Add title
    plt.show()  # Display the plot
