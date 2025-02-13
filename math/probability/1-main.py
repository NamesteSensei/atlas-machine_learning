#!/usr/bin/env python3
"""
Main file to test the Poisson class and PMF method.
"""

from poisson import Poisson

if __name__ == "__main__":
    # Test with data
    data = [2, 3, 4, 3, 2, 3, 4, 5, 3, 4]
    poisson = Poisson(data)
    print("Lambda from data:", poisson.lambtha)
    print("P(2):", poisson.pmf(2))
    print("P(3):", poisson.pmf(3))
    print("P(4):", poisson.pmf(4))

    # Test with lambda (no data)
    poisson2 = Poisson(lambtha=5)
    print("\nLambda (manual):", poisson2.lambtha)
    print("P(5):", poisson2.pmf(5))
    print("P(10):", poisson2.pmf(10))
    print("P(15):", poisson2.pmf(15))
