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
            data: list of data points (optional)
            lambtha: expected occurrences in a time frame (default 1.)
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
            k: number of successes (non-negative integer)

        Returns:
            PMF value for k.
        """
        from math import exp, factorial
        k = int(k)
        if k < 0:
            return 0
        return (exp(-self.lambtha) * self.lambtha ** k) / factorial(k)

    def cdf(self, k):
        """
        Calculate the CDF for a given number of successes (k).

        Args:
            k: number of successes (non-negative integer)

        Returns:
            CDF value for k.
        """
        from math import exp, factorial
        k = int(k)
        if k < 0:
            return 0
        cumulative = 0
        for i in range(k + 1):
            cumulative += (
                (exp(-self.lambtha) * self.lambtha ** i) / factorial(i)
            )
        return cumulative
