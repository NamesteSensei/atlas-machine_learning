#!/usr/bin/env python3
"""Test file for likelihood"""
import numpy as np
likelihood = __import__('0-likelihood').likelihood

if __name__ == '__main__':
    P = np.linspace(0, 1, 11)
    print(likelihood(26, 130, P))
