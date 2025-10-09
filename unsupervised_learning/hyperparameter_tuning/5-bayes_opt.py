#!/usr/bin/env python3
"""
Full Bayesian Optimization loop implementation.
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian Optimization on a 1-D noiseless function.
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """Initialize optimizer."""
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """Expected Improvement acquisition."""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            best = np.min(self.gp.Y)
            imp = best - mu - self.xsi
        else:
            best = np.max(self.gp.Y)
            imp = mu - best - self.xsi

        Z = imp / (sigma + 1e-9)
        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        X_next = self.X_s[np.argmax(EI)].reshape(1,)
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Perform Bayesian optimization for a number of iterations.

        Stops early if a sampled point repeats.

        Returns
        -------
        X_opt : np.ndarray of shape (1,)
            Optimal input.
        Y_opt : np.ndarray of shape (1,)
            Optimal function value.
        """
        for _ in range(iterations):
            X_next, _ = self.acquisition()
            if any(np.allclose(X_next, x) for x in self.gp.X):
                break
            Y_next = self.f(X_next)
            self.gp.update(X_next, Y_next)

        if self.minimize:
            idx = np.argmin(self.gp.Y)
        else:
            idx = np.argmax(self.gp.Y)
        return self.gp.X[idx], self.gp.Y[idx]
