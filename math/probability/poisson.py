#!/usr/bin/env python3
isson Distribution module.
"""


class Poisson:
        """
            Represents a Poisson distribution.
                """

                    def __init__(self, data=None, lambtha=1.):
                            """
                                    Initialize a Poisson distribution.
                                            data: list of data points (optional)
                                                    lambtha: expected number of occurrences in a given time frame (default 1.)
                                                            """
                                                                    if data is None:
                                                                                if lambtha <= 0:
                                                                                                raise ValueError("lambtha must be a positive value")
                                                                                                            self.lambtha = float(lambtha)
                                                                                                                    else:
                                                                                                                                if not isinstance(data, list):
                                                                                                                                                raise TypeError("data must be a list")
                                                                                                                                                            if len(data) < 2:
                                                                                                                                                                            raise ValueError("data must contain multiple values")
                                                                                                                                                                                        self.lambtha = sum(data) / len(data)

                                                                                                                                                                                            def pmf(self, k):
                                                                                                                                                                                                    """
                                                                                                                                                                                                            Calculate the PMF for a given number of successes (k).
                                                                                                                                                                                                                    """
                                                                                                                                                                                                                            from math import exp, factorial
                                                                                                                                                                                                                                    k = int(k)
                                                                                                                                                                                                                                            if k < 0:
                                                                                                                                                                                                                                                        return 0
                                                                                                                                                                                                                                                                return (exp(-self.lambtha) * self.lambtha**k) / factorial(k)

                                                                                                                                                                                                                                                                    def cdf(self, k):
                                                                                                                                                                                                                                                                            """
                                                                                                                                                                                                                                                                                    Calculate the CDF for a given number of successes (k).
                                                                                                                                                                                                                                                                                            """
                                                                                                                                                                                                                                                                                                    from math import exp, factorial
                                                                                                                                                                                                                                                                                                            k = int(k)
                                                                                                                                                                                                                                                                                                                    if k < 0:
                                                                                                                                                                                                                                                                                                                                return 0
                                                                                                                                                                                                                                                                                                                                        cdf_sum = sum((self.lambtha**i) / factorial(i) for i in range(k + 1))
                                                                                                                                                                                                                                                                                                                                                return exp(-self.lambtha) * cdf_sum
                                                                                                                                                                                                                                                                                                                                                

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
