# Matrix and Vector Operations with NumPy
# This code demonstrates the following operations:
# - Dot product of two vectors
# - Matrix multiplication using np.dot and np.matmul
# - Element-wise multiplication of matrices (Hadamard product) using np.multiply

import numpy as np

# Vectors
vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])

# Dot product of vectors
producto_punto = np.dot(vector1, vector2)
print("Dot product of vectors:", producto_punto)

# 2D Matrices
matriz1 = np.array([[1, 2], [3, 4]])
matriz2 = np.array([[5, 6], [7, 8]])

# Matrix multiplication using np.dot
producto_matrices_dot = np.dot(matriz1, matriz2)
print("Matrix multiplication with np.dot:\n", producto_matrices_dot)

# Matrix multiplication using np.matmul
producto_matrices_matmul = np.matmul(matriz1, matriz2)
print("Matrix multiplication with np.matmul:\n", producto_matrices_matmul)

# Element-wise multiplication using np.multiply (Hadamard product)
producto_elemento_a_elemento = np.multiply(matriz1, matriz2)
print("Element-wise multiplication (Hadamard product):\n", producto_elemento_a_elemento)
