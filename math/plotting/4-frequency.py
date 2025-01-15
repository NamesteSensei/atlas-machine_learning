#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for GUI plots
import numpy as np  # Import numpy for generating random data
import matplotlib.pyplot as plt  # Import pyplot for plotting

def frequency():
    """
    Plots a histogram of student grades.
    """
    np.random.seed(5)  # Ensure consistent random values
    student_grades = np.random.normal(68, 15, 50)  # Generate random grades

    plt.hist(student_grades, bins=range(0, 110, 10), edgecolor='black')  # Histogram
    plt.xlabel("Grades")  # Label for the x-axis
    plt.ylabel("Number of Students")  # Label for the y-axis
    plt.title("Project A")  # Title of the graph
    plt.show()  # Display the plot

frequency()  # Call the function to create the histogram
