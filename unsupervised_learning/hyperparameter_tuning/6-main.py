#!/usr/bin/env python3
"""
Main test for 6-bayes_opt.py
Runs Bayesian optimization using GPyOpt on a RandomForest model.
"""

from __future__ import annotations
import matplotlib.pyplot as plt
import numpy as np
import joblib
import os
import GPyOpt
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


# -------------------------------------------------------
# Load dataset
# -------------------------------------------------------
data = load_iris()
X, y = data.data, data.target


# -------------------------------------------------------
# Objective function
# -------------------------------------------------------
def rf_cv(params: np.ndarray) -> float:
    """
    Evaluate RandomForest model with given hyperparameters.

    Returns
    -------
    float
        Negative cross-validation accuracy (for minimization).
    """
    n_est, depth, feat, split, leaf = params[0]
    clf = RandomForestClassifier(
        n_estimators=int(n_est),
        max_depth=int(depth),
        max_features=float(feat),
        min_samples_split=int(split),
        min_samples_leaf=int(leaf),
        random_state=0
    )
    score = cross_val_score(clf, X, y, cv=5).mean()

    # Save checkpoint
    fname = (f"rf_est{int(n_est)}_d{int(depth)}_"
             f"f{feat:.2f}_s{int(split)}_l{int(leaf)}.pkl")
    joblib.dump(clf, fname)

    return -score  # minimize negative accuracy


# -------------------------------------------------------
# Search space
# -------------------------------------------------------
bounds = [
    {'name': 'n_est', 'type': 'discrete', 'domain': (50, 100, 200)},
    {'name': 'depth', 'type': 'discrete', 'domain': (3, 5, 7, 9)},
    {'name': 'feat', 'type': 'continuous', 'domain': (0.3, 1.0)},
    {'name': 'split', 'type': 'discrete', 'domain': (2, 4, 6)},
    {'name': 'leaf', 'type': 'discrete', 'domain': (1, 3, 5)}
]


# -------------------------------------------------------
# Run optimization
# -------------------------------------------------------
if __name__ == '__main__':
    opt = GPyOpt.methods.BayesianOptimization(f=rf_cv, domain=bounds)
    opt.run_optimization(max_iter=30)
    opt.plot_convergence()

    # Save results to file
    with open('bayes_opt.txt', 'w', encoding='utf-8') as f:
        f.write(f"Optimal parameters: {opt.x_opt}\n")
        f.write(f"Optimal value: {opt.fx_opt}\n")

    # Print to terminal
    print("\nBayesian Optimization complete ✅")
    print("Optimal parameters:", opt.x_opt)
    print("Optimal score:", -opt.fx_opt)
    print("\nResults saved to bayes_opt.txt\n")

    # Optional: remove saved model checkpoints to keep clean
    for file in os.listdir():
        if file.endswith('.pkl'):
            os.remove(file)

    plt.show()
