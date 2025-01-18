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

        Args:
            data (list): Optional, data points to estimate the distribution.
            lambtha (float): Expected number of occurrences in a given time frame.
        Raises:
            ValueError: If lambtha is not positive or if data is invalid.
            TypeError: If data is not a list.
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

        Args:
            k (int): Number of successes.
        Returns:
            float: The PMF value for k.
        """
        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285  # Approximation of e
        fact_k = 1
        for i in range(1, k + 1):
            fact_k *= i
        return (self.lambtha ** k * e ** -self.lambtha) / fact_k

    def cdf(self, k):
        """
        Calculate the CDF for a given number of successes (k).

        Args:
            k (int): Number of successes.
        Returns:
            float: The CDF value for k.
        """
        if k < 0:
            return 0
        k = int(k)
        e = 2.7182818285  # Approximation of e
        cumulative = 0
        for i in range(k + 1):
            fact_i = 1
            for j in range(1, i + 1):
                fact_i *= j
            cumulative += (self.lambtha ** i * e ** -self.lambtha) / fact_i
        return cumulative
