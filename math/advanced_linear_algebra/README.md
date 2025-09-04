# Advanced Linear Algebra – Matrix Operations in Pure Python

Welcome to the **Advanced Linear Algebra** module.  
This project focuses on core matrix operations without relying on external libraries like NumPy (except where explicitly allowed).

All logic is implemented from scratch using Python 3.9, strictly following the Holberton School constraints.  
This ensures a deep understanding of how linear algebra operations work under the hood.

---

## 🚀 Tasks Implemented

### `0-determinant.py`
Calculates the determinant of a square matrix using recursive Laplace expansion.

Handles:
- 0×0, 1×1, and 2×2 cases directly
- Any n×n matrix using cofactor expansion

### `1-minor.py`
Returns the minor matrix of a given square matrix.
The minor of an element is the determinant of the matrix left after removing that element’s row and column.

### `2-cofactor.py`
Computes the cofactor matrix by applying a sign pattern to the minor matrix.

### `3-adjugate.py`
Generates the adjugate (transpose of cofactor matrix), a key step in calculating the inverse.

### `4-inverse.py`
Computes the inverse of a matrix using:
1. Determinant
2. Adjugate

Returns `None` for singular (non-invertible) matrices.

### `5-definiteness.py`
Determines whether a matrix is:
- Positive definite
- Positive semi-definite
- Negative definite
- Negative semi-definite
- Indefinite

Uses NumPy (as allowed) for eigenvalue computation.

---

## 📌 Requirements

- Python 3.9
- Ubuntu 20.04
- pycodestyle 2.11.1
- NumPy (only used in task 5)

---

## ✅ Execution

All files are executable. Run the test files using:
```bash
./0-main.py
./1-main.py
# etc.
