#!/usr/bin/env python3
"""
This module plots a stacked bar graph representing the number of fruits
owned by Farrah, Fred, and Felicia.
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plots a stacked bar graph of fruits possessed by individuals.

    - Apples: red
    - Bananas: yellow
    - Oranges: orange (#ff8000)
    - Peaches: peach (#ffe5b4)
    - X-axis: Labeled by names of individuals (Farrah, Fred, Felicia).
    - Y-axis: Labeled 'Quantity of Fruit', ranges 0–80, ticks every 10.
    - Legend included for fruit types.
    - Bars have a width of 0.5 and are stacked from bottom to top.
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))

    # Names and colors
    names = ["Farrah", "Fred", "Felicia"]
    colors = ["red", "yellow", "#ff8000", "#ffe5b4"]

    # Plot stacked bars
    plt.bar(names, fruit[0], width=0.5, color=colors[0], label="Apples")
    plt.bar(names, fruit[1], width=0.5, bottom=fruit[0], color=colors[1],
            label="Bananas")
    plt.bar(names, fruit[2], width=0.5, bottom=fruit[0] + fruit[1],
            color=colors[2], label="Oranges")
    plt.bar(names, fruit[3], width=0.5, bottom=fruit[0] + fruit[1] + fruit[2],
            color=colors[3], label="Peaches")

    # Add labels, title, ticks, and legend
    plt.ylabel("Quantity of Fruit", fontsize='small')
    plt.yticks(range(0, 81, 10))
    plt.title("Number of Fruit per Person", fontsize='small')
    plt.legend(fontsize='small')

    plt.show()
