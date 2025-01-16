#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    """
    Plots a line graph of y = x^3.
    """
    y = np.arange(0, 11) ** 3  # Generate y values (0^3 to 10^3)
    plt.figure(figsize=(6.4, 4.8))  # Set the figure size
    plt.plot(y, 'r-')  # Plot with a solid red line ('r-' = red solid)
    plt.xlim(0, 10)  # Set the x-axis range exactly from 0 to 10
    plt.xlabel("x")  # Label for the x-axis
    plt.ylabel("y")  # Label for the y-axis
    plt.title("Line Graph: y = x^3")  # Add a title to the graph
    plt.show()  # Display the plot
