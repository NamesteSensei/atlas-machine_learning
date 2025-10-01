#!/usr/bin/env python3
import numpy as np
from sklearn.cluster import KMeans

def kmeans(X, k):
    """
    Performs K-means on a dataset

    Args:
        X (np.ndarray): (n, d) dataset
        k (int): number of clusters

    Returns:
        C (np.ndarray): (k, d) array of centroid means
        clss (np.ndarray): (n,) index of the cluster each point belongs to
    """
    kmeans_model = KMeans(n_clusters=k)
    kmeans_model.fit(X)
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_

    return C, clss
