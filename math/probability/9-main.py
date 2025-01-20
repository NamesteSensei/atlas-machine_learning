#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)
print(np.around(n1.cdf(90), 10))  # Should produce the correct desired output

n2 = Normal(mean=70, stddev=10)
print(np.around(n2.cdf(90), 10))  # Should match the desired output
