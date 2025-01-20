#!/usr/bin/env python3
"""
This module defines the Normal class to represent a normal distribution.

The normal distribution is a bell-shaped distribution characterized by
its mean and standard deviation.
"""


class Normal:
    """
    Represents a normal distribution.

    Attributes:
        mean (float): The mean of the distribution.
        stddev (float): The standard deviation of the distribution.
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize the Normal distribution.

        If `data` is provided, the mean and standard deviation are calculated
        from the data. Otherwise, the provided `mean` and `stddev` are used.

        Args:
            data (list): A list of data to estimate the distribution.
            mean (float): The mean of the distribution.
            stddev (float): The standard deviation of the distribution.

        Raises:
            TypeError: If `data` is not a list.
            ValueError: If `data` contains fewer than two values.
            ValueError: If `stddev` is not positive.
        """
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            variance = sum((x - self.mean) ** 2 for x in data) / len(data)
            self.stddev = float(variance ** 0.5)
