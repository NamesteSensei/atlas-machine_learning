#!/usr/bin/env python3
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means.
    
    Parameters:
    - X: np.ndarray of shape (n, d)
    - k: number of clusters
    
    Returns:
    - centroids: np.ndarray of shape (k, d)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    min_vals = np.min(X, axis=0)
    max_vals = np.max(X, axis=0)
    
    return np.random.uniform(min_vals, max_vals, size=(k, X.shape[1]))
