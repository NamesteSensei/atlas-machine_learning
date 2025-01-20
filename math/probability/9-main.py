#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

# Generate data and create the Normal distribution
np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)

# Test CDF using data-driven distribution
print(np.around(n1.cdf(90), 10))  # Expected output: -0.2187964448

# Test CDF using manually set mean and stddev
n2 = Normal(mean=70, stddev=10)
print(np.around(n2.cdf(90), 10))  # Expected output: 0.9872835765
