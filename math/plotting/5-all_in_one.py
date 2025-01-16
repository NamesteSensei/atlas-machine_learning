#!/usr/bin/env python3
"""
This module consolidates all previous plots into one figure with a 3x2 grid.
"""

import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    """
    Plots all previous graphs in one figure.

    - 3x2 grid layout.
    - Axis labels and titles have a font size of x-small.
    - Last plot spans two columns.
    - Figure title: 'All in One'.
    """
    # Data for Task 0
    y0 = np.arange(0, 11) ** 3

    # Data for Task 1
    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    # Data for Task 2
    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    # Data for Task 3
    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    # Data for Task 4
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    # Create the figure and subplots
    fig = plt.figure(figsize=(10, 12))  # Adjust figure size
    fig.suptitle("All in One", fontsize='x-small')

    # Task 0: Line Graph
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(range(11), y0, 'r-')
    ax1.set_title("Task 0: Line Graph", fontsize='x-small')
    ax1.set_xlim(0, 10)

    # Task 1: Scatter Plot
    ax2 = plt.subplot(3, 2, 2)
    ax2.scatter(x1, y1, color='magenta')
    ax2.set_title("Task 1: Scatter Plot", fontsize='x-small')
    ax2.set_xlabel("Height (in)", fontsize='x-small')
    ax2.set_ylabel("Weight (lbs)", fontsize='x-small')

    # Task 2: Change of Scale
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(x2, y2)
    ax3.set_title("Task 2: Exponential Decay of C-14", fontsize='x-small')
    ax3.set_xlabel("Time (years)", fontsize='x-small')
    ax3.set_ylabel("Fraction Remaining", fontsize='x-small')
    ax3.set_yscale("log")
    ax3.set_xlim(0, 28650)

    # Task 3: Two is Better than One
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(x3, y31, 'r--', label="C-14")
    ax4.plot(x3, y32, 'g-', label="Ra-226")
    ax4.set_title("Task 3: Two is Better than One", fontsize='x-small')
    ax4.set_xlabel("Time (years)", fontsize='x-small')
    ax4.set_ylabel("Fraction Remaining", fontsize='x-small')
    ax4.set_xlim(0, 20000)
    ax4.set_ylim(0, 1)
    ax4.legend(fontsize='x-small')

    # Task 4: Frequency Histogram
    ax5 = plt.subplot(3, 2, (5, 6))  # Span two columns
    ax5.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    ax5.set_title("Task 4: Project A", fontsize='x-small')
    ax5.set_xlabel("Grades", fontsize='x-small')
    ax5.set_ylabel("Number of Students", fontsize='x-small')
    ax5.set_xticks(range(0, 101, 10))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout for suptitle
    plt.show()
