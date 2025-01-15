#!/usr/bin/env python3
"""
This script demonstrates slicing specific parts of a matrix.
"""

matrix = [[1, 2, 3, 4, 5, 6],
          [7, 8, 9, 10, 11, 12],
          [13, 14, 15, 16, 17, 18],
          [19, 20, 21, 22, 23, 24]]

mat1 = matrix[1:3]  # Middle two rows: Rows 1 to 2 (not including 3).
mat2 = [row[2:4] for row in matrix]  # Middle two columns: Columns 2 to 3.
mat3 = [row[-3:] for row in matrix[-3:]]  # Bottom-right 3x3 matrix.

print("The middle two rows of the matrix are:\n{}".format(mat1))
print("The middle two columns of the matrix are:\n{}".format(mat2))
print("The bottom-right, square, 3x3 matrix is:\n{}".format(mat3))
