#!/usr/bin/env python3
"""
Poisson Distribution module.
"""


class Poisson:
    """
    Represents a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize a Poisson distribution.
        data: list of data points (optional)
        lambtha: expected number of occurrences in a given time frame
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
        k = int(k)
        if k < 0:
            return 0
        return (self.lambtha ** k * self.exp_neg_lambda()) / self.factorial(k)

    def cdf(self, k):
        """
        Calculate the CDF for a given number of successes (k).
        """
        k = int(k)
        if k < 0:
            return 0
        cumulative = 0
        for i in range(k + 1):
            cumulative += self.pmf(i)
        return cumulative

    def exp_neg_lambda(self):
        """
        Calculate e^(-lambda).
        """
        e = 2.7182818285
        return e ** -self.lambtha

    def factorial(self, num):
        """
        Calculate the factorial of a number.
        """
        if num == 0:
            return 1
        result = 1
        for i in range(1, num + 1):
            result *= i
        return result
