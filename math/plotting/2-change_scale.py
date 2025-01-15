#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for GUI plots
import numpy as np  # Import numpy for numerical calculations
import matplotlib.pyplot as plt  # Import pyplot for plotting

def change_scale():
    """
    Plots an exponential decay curve with a logarithmic y-axis.
    """
    x = np.arange(0, 28651, 5730)  # Time values (0 to 28650 in steps of 5730)
    r = np.log(0.5)  # Decay constant for half-life
    t = 5730  # Half-life of C-14
    y = np.exp((r / t) * x)  # Exponential decay formula

    plt.plot(x, y, 'b-')  # Plot with a solid blue line ('b-' = blue solid)
    plt.xlim(0, 28650)  # Set x-axis range
    plt.yscale('log')  # Use a logarithmic scale for the y-axis
    plt.xlabel("Time (years)")  # Label for the x-axis
    plt.ylabel("Fraction Remaining")  # Label for the y-axis
    plt.title("Exponential Decay of C-14")  # Title of the graph
    plt.show()  # Display the plot

change_scale()  # Call the function to create the plot
