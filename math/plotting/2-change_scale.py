#!/usr/bin/env python3
"""
This script plots the exponential decay of C-14 as a line graph.

The graph includes:
- X-axis labeled 'Time (years)'
- Y-axis labeled 'Fraction Remaining'
- Title: 'Exponential Decay of C-14'
- Logarithmic scale on the y-axis
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots an exponential decay graph of C-14.

    The graph:
    - Shows the fraction remaining of C-14 over time.
    - Uses a logarithmic scale on the y-axis.
    - Includes appropriate labels and a title.
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(x, y, 'r-')
    plt.yscale('log')
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of C-14")
    plt.xlim(0, 28650)
    plt.show()
