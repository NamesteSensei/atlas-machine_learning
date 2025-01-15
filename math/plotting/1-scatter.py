#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for GUI plots
import numpy as np  # Import numpy for generating random data
import matplotlib.pyplot as plt  # Import pyplot for plotting

def scatter():
    """
    Plots a scatter graph of men's height vs. weight.
    """
    mean = [69, 0]  # Average height (69 in) and no weight correlation
    cov = [[15, 8], [8, 15]]  # Covariance matrix for random variations
    np.random.seed(5)  # Ensure consistent random values
    x, y = np.random.multivariate_normal(mean, cov, 2000).T  # Generate data
    y += 180  # Add 180 lbs to weight values

    plt.scatter(x, y, color='magenta')  # Create scatter plot with magenta points
    plt.xlabel("Height (in)")  # Label for the x-axis
    plt.ylabel("Weight (lbs)")  # Label for the y-axis
    plt.title("Men's Height vs Weight")  # Title of the graph
    plt.show()  # Display the plot

scatter()  # Call the function to create the scatter plot
