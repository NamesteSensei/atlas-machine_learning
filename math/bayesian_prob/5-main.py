#!/usr/bin/env python3

import numpy as np
likelihood = __import__('0-likelihood').likelihood

if __name__ == "__main__":
    print(likelihood(55, 100, np.linspace(0, 1, 6)).round(12))
    print(likelihood(33, 70, np.linspace(0, 1, 11)).round(12))
    print(likelihood(22, 50, np.linspace(0, 1, 21)).round(12))
