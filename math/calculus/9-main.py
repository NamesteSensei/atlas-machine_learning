#!/usr/bin/env python3

summation_i_squared = __import__('9-sum_total').summation_i_squared

# Test cases
print(summation_i_squared(5))  # Expected output: 55
print(summation_i_squared(10))  # Expected output: 385
print(summation_i_squared(0))  # Expected output: None
print(summation_i_squared(-5))  # Expected output: None
print(summation_i_squared("a"))  # Expected output: None
