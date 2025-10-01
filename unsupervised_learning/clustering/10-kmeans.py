#!/usr/bin/env python3
"""
K-means clustering using sklearn's KMeans estimator.
"""
import sklearn.cluster


def kmeans(X, k):
    """
    Runs K-means on input data X with k clusters.

    Args:
        X: ndarray of shape (n, d), input samples.
        k: int, number of clusters.

    Returns:
        C: ndarray of shape (k, d), cluster centers.
        clss: ndarray of shape (n,), cluster index per sample.
    """
    model = sklearn.cluster.KMeans(n_clusters=k)
    model.fit(X)
    return model.cluster_centers_, model.labels_
