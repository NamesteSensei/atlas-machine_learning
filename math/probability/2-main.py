#!/usr/bin/env python3

from poisson import Poisson

# Initialize with data
data = [2, 4, 6, 8, 10]
poisson = Poisson(data)

# Test CDF method
print("CDF(3):", poisson.cdf(3))  # Cumulative probability up to 3

# Initialize with lambtha
poisson2 = Poisson(lambtha=5)

# Test CDF method
print("CDF(3):", poisson2.cdf(3))  # Cumulative probability up to 3
print("CDF(5):", poisson2.cdf(5))  # Cumulative probability up to 5
