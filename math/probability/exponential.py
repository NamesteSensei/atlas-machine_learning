#!/usr/bin/env python3
"""
This module defines the Exponential class to represent an distribution.

The exponential distribution models the time between events in a process
where events occur continuously and independently at a constant average rate.
"""


class Exponential:
    """
    Represents an exponential distribution.

    Attributes:
        lambtha (float): The expected number of occurrences in a given
                         time frame.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Exponential distribution.

        If `data` is provided, `lambtha` is calculated as the reciprocal of
        the mean of `data`. Otherwise, `lambtha` is used as provided.

        Args:
            data (list): A list of data to estimate the distribution.
            lambtha (float): The expected number of occurrences in a
                             given time frame.

        Raises:
            TypeError: If `data` is not a list.
            ValueError: If `data` contains fewer than two values.
            ValueError: If `lambtha` is not positive.
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
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        Calculate the PDF value for a given time period.

        The formula is:
        f(x; lambda) = lambda * e^(-lambda * x) for x >= 0
        f(x; lambda) = 0 for x < 0

        Args:
            x (float): The time period.

        Returns:
            float: The probability density function value for x.
        """
        if x < 0:
            return 0
        return self.lambtha * (2.7182818285 ** (-self.lambtha * x))
