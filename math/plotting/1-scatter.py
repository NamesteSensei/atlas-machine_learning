#!/usr/bin/env python3
"""
This module plots a scatter graph of height versus weight.
"""

import numpy as np
import matplotlib.pyplot as plt


def scatter():
    """
    Plots a scatter plot of height vs weight.

    The plot includes:
    - Magenta data points.
    - X-axis labeled 'Height (in)'.
    - Y-axis labeled 'Weight (lbs)'.
    - Title: "Men's Height vs Weight".
    """
    # Generate data
    mean = [69, 0]  # Mean for height and weight
    cov = [[15, 8], [8, 15]]  # Covariance matrix
    np.random.seed(5)  # Set random seed for reproducibility
    x, y = np.random.multivariate_normal(
        mean, cov, 2000
    ).T  # Generate data
    y += 180  # Adjust weight values

    # Create scatter plot
    plt.figure(figsize=(6.4, 4.8))  # Set figure size
    plt.scatter(
        x, y, color='magenta'
    )  # Plot scatter points in magenta
    plt.xlabel("Height (in)")  # Label for x-axis
    plt.ylabel("Weight (lbs)")  # Label for y-axis
    plt.title("Men's Height vs Weight")  # Add a title
    plt.show()  # Display the plot
