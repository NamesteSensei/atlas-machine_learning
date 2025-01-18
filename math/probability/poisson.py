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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes (k).
        k: number of successes
        Returns: PMF value for k
        """
        if k < 0:
            return 0
        k = int(k)
        return (self.lambtha ** k * (2.7182818285 ** -self.lambtha)) / self.factorial(k)

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of successes.
        k: number of successes
        Returns: CDF value for k
        """
        if k < 0:
            return 0

        k = int(k)
        cdf_sum = 0

        for i in range(k + 1):
            cdf_sum += (self.lambtha ** i * (2.7182818285 ** -self.lambtha)) / \
                       self.factorial(i)

        return cdf_sum

    def factorial(self, n):
        """
        Helper method to calculate factorial of n.
        """
        if n == 0:
            return 1
        result = 1
        for i in range(1, n + 1):
            result *= i
        return result
