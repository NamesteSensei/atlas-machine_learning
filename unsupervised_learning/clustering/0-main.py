#!/usr/bin/env python3

import numpy as np
initialize = __import__('0-initialize').initialize

# Create a sample 2D dataset with 100 points in 3 dimensions
np.random.seed(0)
X = np.random.randn(100, 3) * 10 + 50  # 100 points centered around (50,50,50)

# Number of clusters
k = 4

# Call the function
centroids = initialize(X, k)

# Output
print("Centroids:")
print(centroids)
print("Shape:", centroids.shape)
