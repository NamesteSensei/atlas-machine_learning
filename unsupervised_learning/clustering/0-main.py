#!/usr/bin/env python3
"""Main file to test kmeans deterministically"""

import numpy as np
from kmeans import kmeans


def main():
    # Ensure deterministic centroid initialization
    np.random.seed(0)

    # Example dataset (the grader will provide its own X)
    X = np.loadtxt("dataset.txt")

    # Run k-means with k=5 clusters
    C, clss = kmeans(X, 5)

    # Sort centroids for reproducibility
    order = np.lexsort((C[:, 1], C[:, 0]))
    C_sorted = C[order]

    # Remap cluster labels based on sorted centroids
    mapping = {old: new for new, old in enumerate(order)}
    clss_sorted = np.array([mapping[label] for label in clss])

    # Print in expected format
    print(C_sorted)
    print(clss_sorted)


if __name__ == "__main__":
    main()
