
#!/usr/bin/env python3
nential Distribution module.
"""


class Exponential:
        """
            Represents an Exponential distribution.
                """

                    def __init__(self, data=None, lambtha=1.):
                            """
                                    Initialize an Exponential distribution.
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
                                                                                                                                                                        self.lambtha = 1 / (sum(data) / len(data))

                                                                                                                                                                            def pdf(self, x):
                                                                                                                                                                                    """
                                                                                                                                                                                            Calculate the PDF for a given time period (x).
                                                                                                                                                                                                    """
                                                                                                                                                                                                            from math import exp
                                                                                                                                                                                                                    if x < 0:
                                                                                                                                                                                                                                return 0
                                                                                                                                                                                                                                        return self.lambtha * exp(-self.lambtha * x)

                                                                                                                                                                                                                                            def cdf(self, x):
                                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                                            Calculate the CDF for a given time period (x).
                                                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                                                            from math import exp
                                                                                                                                                                                                                                                                                    if x < 0:
                                                                                                                                                                                                                                                                                                return 0
                                                                                                                                                                                                                                                                                                        return 1 - exp(-self.lambtha * x)


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
