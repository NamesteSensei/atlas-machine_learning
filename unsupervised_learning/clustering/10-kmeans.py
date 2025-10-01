#!/usr/bin/env python3
"""
This module implements K-means clustering using scikit-learn's KMeans.

Only the sklearn.cluster module is imported as per project constraints.

Function:
    kmeans(X, k): Performs K-means clustering and returns the centroids
    and the class assignments for each data point.
"""

import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset

    Args:
        X (numpy.ndarray): shape (n, d) containing the dataset
        k (int): number of clusters

    Returns:
        C (numpy.ndarray): shape (k, d) array with centroid means
        clss (numpy.ndarray): shape (n,) array with cluster assignments
    """
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)
    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
