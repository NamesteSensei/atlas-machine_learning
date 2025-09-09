#!/usr/bin/env python3

if __name__ == '__main__':
    cofactor = __import__('2-cofactor').cofactor

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(cofactor(mat1))   # [[1]]
    print(cofactor(mat2))   # [[4, -3], [-2, 1]]
    print(cofactor(mat3))   # [[1, -1], [-1, 1]]
    print(cofactor(mat4))   # [[-12, 36, 0], [-10, -34, 32], [47, -13, -16]]
    try:
        cofactor(mat5)
    except Exception as e:
        print(e)  # matrix must be a list of lists

    try:
        cofactor(mat6)
    except Exception as e:
        print(e)  # matrix must be a non-empty square matrix
