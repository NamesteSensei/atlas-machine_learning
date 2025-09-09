#!/usr/bin/env python3
"""Definiteness of a matrix."""

import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix.

    Args:
        matrix (np.ndarray): square matrix.

    Returns:
        str or None: classification string or None when invalid.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if matrix.size == 0:
        return None

    # Must be symmetric (real case). If not, not a valid input here.
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-8):
        return None

    vals = np.linalg.eigvals(matrix).real
    tol = 1e-8

    if np.all(vals > tol):
        return "Positive definite"
    if np.all(vals >= -tol):
        return "Positive semi-definite"
    if np.all(vals < -tol):
        return "Negative definite"
    if np.all(vals <= tol):
        return "Negative semi-definite"
    if np.any(vals > tol) and np.any(vals < -tol):
        return "Indefinite"
    return None
