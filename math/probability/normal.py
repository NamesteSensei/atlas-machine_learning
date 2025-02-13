class Normal:
    def __init__(self, data=None, mean=0., stddev=1.):
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
            self.stddev = float((sum([(x - self.mean) ** 2 for x in data]) / len(data)) ** 0.5)

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
        coeff = 1 / (self.stddev * (2 * 3.1415926536) ** 0.5)
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
        t = 1 / (1 + 0.3275911 * abs(z))
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        erf = 1 - (a1 * t + a2 * t**2 + a3 * t**3 + a4 * t**4 + a5 * t**5) * (2.7182818285 ** (-z**2))
        if z < 0:
            erf = -erf
        return 0.5 * (1 + erf)
