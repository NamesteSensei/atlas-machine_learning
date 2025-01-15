#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for GUI plots
import numpy as np  # Import numpy for numerical calculations
import matplotlib.pyplot as plt  # Import pyplot for plotting

def two():
    """
    Plots two exponential decay curves on the same graph.
    """
    x = np.arange(0, 21000, 1000)  # Time values (0 to 20000 in steps of 1000)
    r = np.log(0.5)  # Decay constant for half-life
    t1 = 5730  # Half-life of C-14
    t2 = 1600  # Half-life of Ra-226
    y1 = np.exp((r / t1) * x)  # Exponential decay for C-14
    y2 = np.exp((r / t2) * x)  # Exponential decay for Ra-226

    plt.plot(x, y1, 'r--', label="C-14")  # Red dashed line for C-14
    plt.plot(x, y2, 'g-', label="Ra-226")  # Green solid line for Ra-226
    plt.legend(loc="upper right")  # Add legend in the upper right corner
    plt.xlabel("Time (years)")  # Label for the x-axis
    plt.ylabel("Fraction Remaining")  # Label for the y-axis
    plt.title("Exponential Decay of Radioactive Elements")  # Title
    plt.xlim(0, 20000)  # Set x-axis range
    plt.ylim(0, 1)  # Set y-axis range
    plt.show()  # Display the plot

two()  # Call the function to create the plot
