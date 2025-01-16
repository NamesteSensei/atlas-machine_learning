#!/usr/bin/env python3
"""
This module plots an exponential decay graph of C-14.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots a line graph showing the exponential decay of C-14.

    The graph includes:
    - X-axis labeled 'Time (years)'.
    - Y-axis labeled 'Fraction Remaining'.
    - Title: 'Exponential Decay of C-14'.
    - Y-axis is logarithmically scaled.
    - X-axis ranges from 0 to 28,650.
    """
    # Generate data
    x = np.arange(0, 28651, 5730)  # Time points (in years)
    r = np.log(0.5)  # Natural log of 0.5 (decay constant)
    t = 5730  # Half-life of C-14
    y = np.exp((r / t) * x)  # Calculate fraction remaining

    # Create line graph
    plt.figure(figsize=(6.4, 4.8))  # Set figure size
    plt.plot(x, y, 'r-')  # Plot with a solid red line
    plt.xscale('linear')  # X-axis is linear (default)
    plt.yscale('log')  # Y-axis is logarithmic
    plt.xlabel("Time (years)")  # Label for x-axis
    plt.ylabel("Fraction Remaining")  # Label for y-axis
    plt.title("Exponential Decay of C-14")  # Add a title
    plt.show()  # Display the plot
