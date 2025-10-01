#!/usr/bin/env python3
"""
Gaussian Mixture Model clustering using sklearn.mixture.GaussianMixture.
"""
import sklearn.mixture


def gmm(X, k):
    """
    Fits a GMM to data X with k components.

    Args:
        X: ndarray of shape (n, d), input samples.
        k: int, number of mixture components.

    Returns:
        pi: ndarray of shape (k,), cluster priors.
        m: ndarray of shape (k, d), cluster centers.
        S: ndarray of shape (k, d, d), covariances.
        clss: ndarray of shape (n,), predicted labels.
        bic: ndarray of shape (k,), BIC scores from kmin to kmax.
    """
    g = sklearn.mixture.GaussianMixture(n_components=k)
    g.fit(X)
    pi = g.weights_
    m = g.means_
    S = g.covariances_
    clss = g.predict(X)
    bic = g.bic(X)
    return pi, m, S, clss, bic
