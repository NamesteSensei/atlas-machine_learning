
rmal Distribution module.
"""


class Normal:
        """
            Represents a Normal distribution.
                """

                    def __init__(self, data=None, mean=0., stddev=1.):
                            """
                                    Initialize a Normal distribution.
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
                                                                                                                                                                                    self.mean = sum(data) / len(data)
                                                                                                                                                                                                self.stddev = (sum((x - self.mean)**2 for x in data) / len(data))**0.5

                                                                                                                                                                                                    def z_score(self, x):
                                                                                                                                                                                                            """
                                                                                                                                                                                                                    Calculate the z-score for a given x-value.
                                                                                                                                                                                                                            """
                                                                                                                                                                                                                                    return (x - self.mean) / self.stddev

                                                                                                                                                                                                                                        def x_value(self, z):
                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                        Calculate the x-value for a given z-score.
                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                        return z * self.stddev + self.mean

                                                                                                                                                                                                                                                                            def pdf(self, x):
                                                                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                                                                            Calculate the PDF for a given x-value.
                                                                                                                                                                                                                                                                                                    """
                                                                                                                                                                                                                                                                                                            from math import exp, pi, sqrt
                                                                                                                                                                                                                                                                                                                    return (1 / (self.stddev * sqrt(2 * pi))) * exp(-0.5 * ((x - self.mean) / self.stddev)**2)

                                                                                                                                                                                                                                                                                                                        def cdf(self, x):
                                                                                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                                                                                        Calculate the CDF for a given x-value.
                                                                                                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                                                                                                        from math import erf, sqrt
                                                                                                                                                                                                                                                                                                                                                                return 0.5 * (1 + erf((x - self.mean) / (self.stddev * sqrt(2))))

