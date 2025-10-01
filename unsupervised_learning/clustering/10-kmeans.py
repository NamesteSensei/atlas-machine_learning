#!/usr/bin/env python3
"""
This module implements K-means clustering using scikit-learn's KMeans.

Function:
    kmeans(X, k): Fits K-means to the data and returns cluster centroids
    and index labels for each data point.
"""

import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k):
    """
    Performs K-means on a dataset

    Args:
        X (np.ndarray): shape (n, d) dataset
        k (int): number of clusters

    Returns:
        C (np.ndarray): shape (k, d), centroid coordinates
        clss (np.ndarray): shape (n,), index of the cluster each point belongs to
    """
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(X)
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_

    return C, clss
