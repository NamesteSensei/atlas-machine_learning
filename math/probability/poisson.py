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
        if k < 0:
            return 0
        k = int(k)
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        pmf = (self.lambtha ** k) * (self.exp_neg_lambtha()) / factorial
        return pmf

    def cdf(self, k):
        """
        Calculate the CDF for a given number of successes (k).
        """
        if k < 0:
            return 0
        k = int(k)
        cumulative = 0
        for i in range(k + 1):
            # Manual factorial calculation
            factorial = 1
            for j in range(1, i + 1):
                factorial *= j

            cumulative += (
                (self.lambtha ** i * self.exp_neg_lambtha()) / factorial
            )
        return cumulative

    def exp_neg_lambtha(self):
        """
        Helper function to calculate e^(-lambtha) using approximation.
        """
        terms = 20  # Number of terms for the approximation
        result = 1
        power = 1
        factorial = 1
        for n in range(1, terms + 1):
            power *= -self.lambtha
            factorial *= n
            result += power / factorial
        return result
