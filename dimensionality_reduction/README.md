# Dimensionality Reduction Project

## 📚 Description

This project implements dimensionality reduction techniques using **Principal Component Analysis (PCA)**. It is part of the "Machine Learning - Track 4" curriculum at Holberton School and focuses on reducing the number of features in high-dimensional datasets while preserving essential information.

We implement two PCA-based approaches:
1. Reducing dimensions while preserving a target variance
2. Reducing to a specified number of dimensions

---

## 🎯 Learning Objectives

- Understand the concept and purpose of **dimensionality reduction**
- Implement **eigendecomposition** and **singular value decomposition (SVD)**
- Apply **Principal Component Analysis (PCA)**
- Differentiate between **linear and non-linear dimensionality reduction**
- Get introduced to **manifolds** and **t-SNE**

---

## 🧠 Concepts Covered

- Covariance matrices
- Eigenvalues and eigenvectors
- Cumulative variance preservation
- Projection of high-dimensional data
- `numpy.linalg.eigh` for symmetric matrices
- Tradeoffs between EIG and SVD performance

---

## 🛠️ Files

| File        | Description |
|-------------|-------------|
| `0-pca.py`  | Implements PCA preserving a specified fraction of variance |
| `1-pca.py`  | Implements PCA reducing to a fixed number of dimensions |
| `0-main.py` | Script to test `0-pca.py` with synthetic data |
| `1-main.py` | Script to test `1-pca.py` with MNIST dataset |
| `mnist2500_X.txt` | Input dataset (2500 MNIST samples) |
| `mnist2500_labels.txt` | Corresponding labels (used optionally for validation) |

---

## 🚀 Usage

Make sure your scripts are executable and run them directly:

```bash
chmod +x 0-main.py 1-main.py
./0-main.py
./1-main.py

