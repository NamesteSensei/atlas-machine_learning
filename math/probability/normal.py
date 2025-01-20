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
        Calculate the PDF value for a given x-value.

        The formula is:
        f(x; mean, stddev) = (1 / sqrt(2 * pi * stddev^2)) *
                             e^(-(x - mean)^2 / (2 * stddev^2))

        Args:
            x (float): The x-value.

        Returns:
            float: The PDF value for x.
        """
        coeff = 1 / (self.stddev * (2 * 3.14159265359) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        return coeff * (2.7182818285 ** exponent)

    def cdf(self, x):
        """
        Calculate the CDF value for a given x-value.

        The formula is:
        CDF(x) = 0.5 * [1 + erf((x - mean) / (stddev * sqrt(2)))]

        Args:
            x (float): The x-value.

        Returns:
            float: The CDF value for x.
        """
        z = (x - self.mean) / (self.stddev * (2 ** 0.5))
        erf = (2 / (3.14159265359 ** 0.5)) * (
            z - (z ** 3) / 3 + (z ** 5) / 10 - (z ** 7) / 42
        )
        return 0.5 * (1 + erf)
