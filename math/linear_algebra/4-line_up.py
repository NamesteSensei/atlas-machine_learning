#!/usr/bin/env python3
"""
This module contains a function to add two arrays element-wise.
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list): First array.
        arr2 (list): Second array.

    Returns:
        list or None: A new array with elements added element-wise, or None if
        the arrays have different lengths.
    """
    if len(arr1) != len(arr2):
        return None
    return [a + b for a, b in zip(arr1, arr2)]
