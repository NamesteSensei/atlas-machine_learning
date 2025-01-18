#!/usr/bin/env python3
"""
Binomial Distribution module.
"""


class Binomial:
        """
            Represents a Binomial distribution.
                """

                    def __init__(self, data=None, n=1, p=0.5):
                                """
                                        Initialize a Binomial distribution.
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
                                                                                                                                                                                                                                        variance = sum((x - mean)**2 for x in data) / len(data)
                                                                                                                                                                                                                                                    self.p = 1 - (variance / mean)
                                                                                                                                                                                                                                                                self.n = round(mean / self.p)

                                                                                                                                                                                                                                                                    def pmf(self, k):
                                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                                        Calculate the PMF for a given number of successes (k).
                                                                                                                                                                                                                                                                                                """
                                                                                                                                                                                                                                                                                                        from math import comb
                                                                                                                                                                                                                                                                                                                k = int(k)
                                                                                                                                                                                                                                                                                                                        if k < 0 or k > self.n:
                                                                                                                                                                                                                                                                                                                                        return 0
                                                                                                                                                                                                                                                                                                                                            return comb(self.n, k) * (self.p**k) * ((1 - self.p)**(self.n - k))

                                                                                                                                                                                                                                                                                                                                            def cdf(self, k):
                                                                                                                                                                                                                                                                                                                                                        """
                                                                                                                                                                                                                                                                                                                                                                Calculate the CDF for a given number of successes (k).
                                                                                                                                                                                                                                                                                                                                                                        """
                                                                                                                                                                                                                                                                                                                                                                                return sum(self.pmf(i) for i in range(int(k) + 1))
                                                                                                                                                                                                                                                                                                                    
