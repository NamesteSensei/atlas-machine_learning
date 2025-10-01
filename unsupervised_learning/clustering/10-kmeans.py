#!/usr/bin/env python3
"""
This module implements K-means clustering using scikit-learn's KMeans.

Only the sklearn.cluster module is imported as per project constraints.

Function:
    kmeans(X, k): Fits K-means to the data and returns cluster centroids
    and index labels for each data point.
"""

import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset.

    Args:
        X (numpy.ndarray): shape (n, d) dataset.
        k (int): number of clusters.

    Returns:
        C (numpy.ndarray): shape (k, d), centroid coordinates.
        clss (numpy.ndarray): shape (n,), index of the cluster each point belongs to.
    """
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)
    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
