obility Distributions in Python

## Overview
This project focuses on implementing and understanding various probability distributions using Python. Each distribution is implemented as a Python class with methods for calculating key properties like the Probability Mass Function (PMF) and Cumulative Distribution Function (CDF).

---

## Files and Directories

### **1. poisson.py**
This file contains the implementation of the `Poisson` class. It represents a Poisson distribution and provides methods to calculate:
- The Probability Mass Function (PMF)
- The Cumulative Distribution Function (CDF)

**Key Concepts for Poisson Distribution**:
- Used for modeling the number of occurrences of an event in a fixed interval.
- `lambtha`: The average number of occurrences.

### **2. 0-main.py**
This is a test file to validate the implementation of the `Poisson` class. It initializes the class and verifies the calculation of `lambtha`.

### **3. math/probability**
This directory contains all implementation files for the project.

---

## Concepts Covered

### **1. Probability**
- **Definition**: The measure of the likelihood of an event occurring.
- **Key Terms**:
  - Independence
  - Disjoint Events
  - Union & Intersection

### **2. Probability Distributions**
- **Probability Mass Function (PMF)**: The probability that a discrete random variable is equal to a specific value.
- **Cumulative Distribution Function (CDF)**: The probability that a random variable is less than or equal to a specific value.

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone <repository_url>
!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph representing the number of fruit various people possess.
    
    The x-axis is labeled with the names of the people.
    The y-axis is labeled 'Quantity of Fruit'.
    The y-axis ranges from 0 to 80 with ticks every 10 units.
    The title of the graph is 'Number of Fruit per Person'.
    Each fruit is represented by a specific color:
        - apples = red
        - bananas = yellow
        - oranges = orange (#ff8000)
        - peaches = peach (#ffe5b4)
    A legend is used to indicate which fruit is represented by each color.
    The bars are stacked in the same order as the rows of fruit, from bottom to top.
    The bars have a width of 0.5.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # Plotting the stacked bar graph
    labels = ['Farrah', 'Fred', 'Felicia']
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    plt.bar(labels, apples, width=0.5, color='red', label='apples')
    plt.bar(labels, bananas, width=0.5, bottom=apples, color='yellow', label='bananas')
    plt.bar(labels, oranges, width=0.5, bottom=apples + bananas, color='#ff8000', label='oranges')
    plt.bar(labels, peaches, width=0.5, bottom=apples + bananas + oranges, color='#ffe5b4', label='peaches')

    plt.xlabel('Person')
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.ylim(0, 80)
    plt.yticks(range(0, 81, 10))
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":

