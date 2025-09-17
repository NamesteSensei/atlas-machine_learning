#!/usr/bin/env python3
"""
Early Stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if gradient descent should be stopped early.

    Parameters
    ----------
    cost : float
        Current validation cost.
    opt_cost : float
        Lowest recorded validation cost.
    threshold : float
        Minimum threshold for significant improvement.
    patience : int
        Number of epochs allowed without improvement.
    count : int
        Number of epochs since threshold was met.

    Returns
    -------
    (bool, int)
        Boolean indicating if training should stop early,
        and the updated patience count.
    """
    if opt_cost - cost > threshold:
        return False, 0

    count += 1
    if count >= patience:
        return True, count

    return False, count
