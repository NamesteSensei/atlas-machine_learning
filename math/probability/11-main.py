#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('P(30):', b1.pmf(30))  # Expected: ~0.11412829839570347

b2 = Binomial(n=50, p=0.6)
print('P(30):', b2.pmf(30))  # Expected: ~0.114558552829524
