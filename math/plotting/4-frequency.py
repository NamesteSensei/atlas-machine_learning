#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plots a histogram of student scores for a project.
    
    The x-axis is labeled 'Grades'.
    The y-axis is labeled 'Number of Students'.
    The x-axis has bins every 10 units.
    The title of the histogram is 'Project A'.
    The bars are outlined in black.
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Plotting the histogram
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')
    plt.xlabel('Grades')
    plt.ylabel('Number of Students')
    plt.title('Project A')
    plt.show()


# Example usage
if __name__ == "__main__":
    frequency()
