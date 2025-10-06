#!/usr/bin/env python3

import numpy as np
absorbing = __import__('2-absorbing').absorbing

if __name__ == "__main__":
    a = np.eye(2)
    b = np.array([[0.6, 0.4],
                  [0.3, 0.7]])
    c = np.array([[0.25, 0.2, 0.25, 0.3],
                  [0.2, 0.3, 0.2, 0.3],
                  [0.25, 0.25, 0.4, 0.1],
                  [0.3, 0.3, 0.1, 0.3]])
    print(absorbing(a))
    print(absorbing(b))
    print(absorbing(c))
