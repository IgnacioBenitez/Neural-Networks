import numpy as np

vector1 = np.array([1, 2, 3])
vector2 = np.array([4, 5, 6])


producto_punto = np.dot(vector1, vector2)
print("Producto punto de vectores:", producto_punto)

# Multiplicación de Matrices (2D)
matriz1 = np.array([[1, 2], [3, 4]])
matriz2 = np.array([[5, 6], [7, 8]])

# Producto de matrices con np.dot
producto_matrices_dot = np.dot(matriz1, matriz2)
print("Multiplicación de matrices con np.dot:\n", producto_matrices_dot)

# Producto de matrices con np.matmul
producto_matrices_matmul = np.matmul(matriz1, matriz2)
print("Multiplicación de matrices con np.matmul:\n", producto_matrices_matmul)

# Multiplicación elemento a elemento con np.multiply
producto_elemento_a_elemento = np.multiply(matriz1, matriz2)
print("Multiplicación elemento a elemento (Hadamard):\n", producto_elemento_a_elemento)
