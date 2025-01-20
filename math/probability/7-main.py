#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

# Create data and a Normal distribution
np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
n1 = Normal(data)

# PDF for a specific value
print("PDF at x=70:", n1.pdf(70))  # Should compute using mean and stddev from data

# Create another Normal distribution with specific mean and stddev
n2 = Normal(mean=70, stddev=10)
print("PDF at x=70:", n2.pdf(70))  # Should compute using mean=70 and stddev=10
