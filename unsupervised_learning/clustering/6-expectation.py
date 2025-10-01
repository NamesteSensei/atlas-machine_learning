#!/usr/bin/env python3
import numpy as np
pdf = __import__('5-pdf').pdf

def expectation(X, pi, m, S):
    try:
        if not isinstance(X, np.ndarray) or X.ndim != 2:
            return None, None
        if not isinstance(pi, np.ndarray) or pi.ndim != 1:
            return None, None
        if not isinstance(m, np.ndarray) or m.ndim != 2:
            return None, None
        if not isinstance(S, np.ndarray) or S.ndim != 3:
            return None, None
        n, d = X.shape
        k = pi.shape[0]
        if m.shape != (k, d) or S.shape != (k, d, d):
            return None, None

        # Compute weighted likelihoods
        like = np.array([pi[i] * pdf(X, m[i], S[i]) for i in range(k)])
        tot = np.sum(like, axis=0)

        # Normalize responsibilities
        g = like / tot

        # Log likelihood
        l = np.sum(np.log(tot))

        return g, l
    except Exception:
        return None, None
