#!/usr/bin/env python3
"""
This module defines the Binomial class to represent a binomial distribution.

The binomial distribution models the number of successes in n independent
Bernoulli trials with probability p of success.
"""


class Binomial:
    """
    Represents a binomial distribution.

    Attributes:
        n (int): The number of Bernoulli trials.
        p (float): The probability of a success.
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialize the Binomial distribution.

        If `data` is provided, the number of trials (n) and the probability
        of success (p) are calculated from the data.

        Args:
            data (list): A list of data to estimate the distribution.
            n (int): The number of Bernoulli trials.
            p (float): The probability of success.

        Raises:
            TypeError: If `data` is not a list.
            ValueError: If `data` contains fewer than two values.
            ValueError: If `n` is not a positive integer.
            ValueError: If `p` is not in the range (0, 1).
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)
            self.p = mean / self.n

    def factorial(self, x):
        """
        Calculate the factorial of a number.

        Args:
            x (int): The number to calculate the factorial of.

        Returns:
            int: The factorial of x.
        """
        if x == 0 or x == 1:
            return 1
        return x * self.factorial(x - 1)

    def pmf(self, k):
        """
        Calculate the PMF value for a given number of successes.

        Args:
            k (int): The number of successes.

        Returns:
            float: The PMF value for k, or 0 if k is out of range.
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:
            return 0
        combination = self.factorial(self.n) / (
            self.factorial(k) * self.factorial(self.n - k)
        )
        return combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))
