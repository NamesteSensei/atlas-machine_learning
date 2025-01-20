#!/usr/bin/env python3

import numpy as np
Normal = __import__('normal').Normal

np.random.seed(0)
data = np.random.normal(70, 10, 100).tolist()
norm = Normal(data)

# Calculate z-scores for some data points
for i in range(5):
    print(np.around(norm.z_score(data[i]), 10))

# Calculate x-values for some z-scores
z_scores = [-1.5, -0.5, 0, 0.5, 1.5]
for z in z_scores:
    print(np.around(norm.x_value(z), 10))
