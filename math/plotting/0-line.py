#!/usr/bin/env python3
"""
This module contains a function to plot a cubic line graph.

The line graph is plotted for y = x^3 where x ranges from 0 to 10.
The plot displays a solid red line with an appropriate axis range.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Plots a line graph of y = x^3.

    The graph displays:
    - A solid red line representing y = x^3.
    - The x-axis ranges from 0 to 10.
    """
    y = np.arange(0, 11) ** 3  # Generate y values (0^3 to 10^3)
    plt.figure(figsize=(6.4, 4.8))  # Set the figure size
    plt.plot(y, 'r-')  # Plot with a solid red line ('r-' = red solid)
    plt.xlim(0, 10)  # Set the x-axis range exactly from 0 to 10
    plt.show()  # Display the plot
