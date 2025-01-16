#!/usr/bin/env python3
"""
This module plots a histogram of student grades for a project.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student grades.

    The histogram:
    - Has bins every 10 units on the x-axis.
    - X-axis is labeled 'Grades'.
    - Y-axis is labeled 'Number of Students'.
    - Title: 'Project A'.
    - Bars are outlined in black.
    """
    np.random.seed(5)  # Seed for reproducibility
    student_grades = np.random.normal(68, 15, 50)  # Generate student grades

    # Create histogram
    plt.figure(figsize=(6.4, 4.8))  # Set figure size
    plt.hist(
        student_grades,
        bins=range(0, 101, 10),  # Bins every 10 units
        edgecolor='black'  # Outline bars in black
    )
    plt.xlabel("Grades")  # Label x-axis
    plt.ylabel("Number of Students")  # Label y-axis
    plt.title("Project A")  # Add title
    plt.xticks(range(0, 101, 10))  # Set x-axis ticks every 10
    plt.show()
