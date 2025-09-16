#!/usr/bin/env python3
"""Test file for intersection"""
import numpy as np
intersection = __import__('1-intersection').intersection

if __name__ == '__main__':
    P = np.linspace(0, 1, 11)
    Pr = np.ones(11) / 11
    print(intersection(26, 130, P, Pr))
