#!/usr/bin/env python3

if __name__ == '__main__':
    definiteness = __import__('5-definiteness').definiteness
    import numpy as np

    mat1 = np.array([[5, 1], [1, 1]])
    mat2 = np.array([[2, 4], [4, 8]])
    mat3 = np.array([[-1, 1], [1, -1]])
    mat4 = np.array([[-2, 4], [4, -9]])
    mat5 = np.array([[1, 2], [2, 1]])
    mat6 = np.array([])
    mat7 = np.array([[1, 2, 3], [4, 5, 6]])
    mat8 = [[1, 2], [1, 2]]

    print(definiteness(mat1))  # Positive definite
    print(definiteness(mat2))  # Positive semi-definite
    print(definiteness(mat3))  # Negative semi-definite
    print(definiteness(mat4))  # Negative definite
    print(definiteness(mat5))  # Indefinite
    print(definiteness(mat6))  # None
    print(definiteness(mat7))  # None
    try:
        definiteness(mat8)
    except Exception as e:
        print(e)  # matrix must be a numpy.ndarray
