#!/usr/bin/env python3
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
