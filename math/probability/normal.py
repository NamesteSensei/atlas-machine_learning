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

    def pdf(self, x):
        """
        Calculate the PDF value for a given data point.

        The formula is:
        f(x; mean, stddev) = (1 / sqrt(2 * pi * stddev^2)) *
                             e^(-(x - mean)^2 / (2 * stddev^2))

        Args:
            x (float): The data point.

        Returns:
            float: The PDF value for the given x.
        """
        coeff = 1 / (self.stddev * (2 * 3.14159265359) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        return coeff * (2.7182818285 ** exponent)

    def z_score(self, x):
        """
        Calculate the z-score of a given data point.

        The formula is:
        z = (x - mean) / stddev

        Args:
            x (float): The data point.

        Returns:
            float: The z-score of x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the data point (x-value) corresponding to a given z-score.

        The formula is:
        x = mean + z * stddev

        Args:
            z (float): The z-score.

        Returns:
            float: The corresponding x-value.
        """
        return self.mean + z * self.stddev
