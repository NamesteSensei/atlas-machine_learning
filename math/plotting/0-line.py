#!/usr/bin/env python3
"""
Module to plot a line graph of y = x^3.
"""

import matplotlib  # Import matplotlib for setting the backend
import numpy as np  # Import numpy for numerical calculations
import matplotlib.pyplot as plt  # Import pyplot for plotting

# Set the backend for matplotlib before any other matplotlib code
matplotlib.use('TkAgg')


def line():
    """
    Plots a line graph of y = x^3.
    """
    y = np.arange(0, 11) ** 3  # Create y values (0^3 to 10^3)
    plt.plot(y, 'r-')  # Plot with a solid red line ('r-' = red solid)
    plt.xlim(0, 10)  # Set the x-axis range from 0 to 10
    plt.xlabel("x")  # Label for the x-axis
    plt.ylabel("y")  # Label for the y-axis
    plt.title("Line Graph: y = x^3")  # Title of the graph
    plt.show()  # Display the plot


if __name__ == "__main__":
    line()  # Call the function to create the plot
