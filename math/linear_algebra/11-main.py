#!/usr/bin/env python3

import numpy as np
np_transpose = __import__('11-the_western_exchange').np_transpose

mat1 = np.array([1, 2, 3, 4, 5, 6])  # 1D array
mat2 = np.array([])  # Empty array
mat3 = np.array([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
                 [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]])  # 3D array

print(np_transpose(mat1))  # Expected: [1 2 3 4 5 6]
print(mat1)  # Ensure mat1 is not modified
print(np_transpose(mat2))  # Expected: []
print(mat2)  # Ensure mat2 is not modified
print(np_transpose(mat3))  # Check transpose of 3D array
print(mat3)  # Ensure mat3 is not modified
