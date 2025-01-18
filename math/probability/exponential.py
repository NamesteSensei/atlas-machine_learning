#!/usr/bin/env python3
"""
This module defines the Exponential class to represent an exponential distribution.
"""


class Exponential:
    """
    Represents an exponential distribution.
    """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize the Exponential distribution.

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
            self.lambtha = float(1 / (sum(data) / len(data)))
