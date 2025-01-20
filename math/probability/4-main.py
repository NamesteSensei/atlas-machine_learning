#!/usr/bin/env python3

Exponential = __import__('exponential').Exponential

# Example 1: Using data to calculate lambtha
data = [2, 4, 6, 8, 10]
exp = Exponential(data)

# PDF for x = 3
print("PDF for x=3:", exp.pdf(3))  # Should compute based on lambtha from data

# Example 2: Specifying lambtha directly
exp2 = Exponential(lambtha=2)

# PDF for x = 1
print("PDF for x=1:", exp2.pdf(1))  # Should compute with lambtha=2
