#!/usr/bin/env python3

"""
This module contains a function to plot x ↦ y as a line graph with specified requirements.
"""

import numpy as np
import matplotlib.pyplot as plt


def change_scale():
    """
    Plots x ↦ y as a line graph with specified requirements.
    
    The x-axis is labeled 'Time (years)'.
    The y-axis is labeled 'Fraction Remaining'.
    The title of the graph is 'Exponential Decay of C-14'.
    The y-axis is logarithmically scaled.
    The x-axis ranges from 0 to 28650.
    """
    x = np.arange(0, 28651, 5730)
    r = np.log(0.5)
    t = 5730
    y = np.exp((r / t) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plotting the graph
    plt.plot(x, y)
    plt.xlabel('Time (years)')
    plt.ylabel('Fraction Remaining')
    plt.title('Exponential Decay of C-14')
    plt.yscale('log')
    plt.xlim(0, 28650)
    plt.show()


# Example usage
if __name__ == "__main__":
    change_scale()
