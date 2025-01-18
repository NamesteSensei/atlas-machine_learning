#!/usr/bin/env python3
"""
This module defines the Poisson class to represent a Poisson distribution.
"""


class Poisson:
    """
    Represents a Poisson distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Poisson distribution.

        Args:
            data (list): A list of data to estimate the distribution.
            lambtha (float): The expected number of occurrences in a
                             given time frame.

        Raises:
            TypeError: If data is not a list.
            ValueError: If data contains fewer than two values.
            ValueError: If lambtha is not positive.
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
        Calculate the PMF value for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k.
        """
        if k < 0:
            return 0
        k = int(k)
        factorial = 1
        for i in range(1, k + 1):
            factorial *= i
        pmf_value = ((self.lambtha ** k) *
                     (2.7182818285 ** -self.lambtha) / factorial)
        return pmf_value

    def cdf(self, k):
        """
        Calculate the CDF value for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The CDF value for k.
        """
        if k < 0:
            return 0
        k = int(k)
        cumulative = 0
        for i in range(0, k + 1):
            factorial = 1
            for j in range(1, i + 1):
                factorial *= j
            cumulative += ((self.lambtha ** i) *
                           (2.7182818285 ** -self.lambtha) / factorial)
        return cumulative
