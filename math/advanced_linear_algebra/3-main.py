#!/usr/bin/env python3

if __name__ == '__main__':
    adjugate = __import__('3-adjugate').adjugate

    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(adjugate(mat1))  # [[1]]
    print(adjugate(mat2))  # [[4, -2], [-3, 1]]
    print(adjugate(mat3))  # [[1, -1], [-1, 1]]
    print(adjugate(mat4))  # [[-12, -10, 47], [36, -34, -13], [0, 32, -16]]
    try:
        adjugate(mat5)
    except Exception as e:
        print(e)  # matrix must be a list of lists

    try:
        adjugate(mat6)
    except Exception as e:
        print(e)  # matrix must be a non-empty square matrix
