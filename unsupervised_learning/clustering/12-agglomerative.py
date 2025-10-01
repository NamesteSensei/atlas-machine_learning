#!/usr/bin/env python3
"""
Agglomerative clustering using Ward linkage with dendrogram saved to file.
"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Runs agglomerative clustering on X using Ward linkage.

    Args:
        X: ndarray of shape (n, d), input samples.
        dist: float, max cophenetic distance for clusters.

    Returns:
        clss: ndarray of shape (n,), cluster label per sample.
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')

    clss = scipy.cluster.hierarchy.fcluster(
        Z, dist, criterion='distance'
    )

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(Z, color_threshold=dist)
    plt.axhline(y=dist, color='k', linestyle='--')
    plt.title('Agglomerative Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.tight_layout()
    plt.savefig('dendrogram.png')

    return clss
