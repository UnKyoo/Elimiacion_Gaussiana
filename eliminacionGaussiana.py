#   Codigo que implementa el esquema numerico 
#   del metodo de eliminación Gaussiana para
#   resolver sistemas de ecuaciones
#
#           Autor:
#   Gilbert Alexander Mendez Cervera
#   mendezgilbert222304@outlook.com
#   Version 1.01 : 25/02/2025
#

import numpy as np

def gauss_elimination(A, b): #DEfinicion del metodo par ael calculo de matriz añadida
    n = len(b)
    for i in range(n):
        # Pivoteo parcial
        max_row = i + np.argmax(np.abs(A[i:, i]))
        if max_row != i:
            A[[i, max_row]] = A[[max_row, i]]
            b[[i, max_row]] = b[[max_row, i]]
        
        # Eliminación hacia adelante
        for j in range(i+1, n):
            factor = A[j][i] / A[i][i]
            A[j, i:] -= factor * A[i, i:]
            b[j] -= factor * b[i]
    
    # Sustitución regresiva
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
    return x

# Definición del sistema de ecuaciones
# La matriz A se incluye por renglones
# El vector b se incluye por columnas
#JERCICIO 1
A = np.array([[3, 2, -1, 4], 
              [5, -3, 2, -1], 
              [-1, 4, -2, 3],
              [2, -1, 3, 5]], dtype=float)
b = np.array([10, 5, -3, 8], dtype=float)


""" #EJERCICIO 2
A = np.array([[6, -2, 3, -1, 2], 
              [-3, 5, -2, 4, -1], 
              [4, 3, 7, -5, 3],
              [-2, 6, -3, 1, -4],
              [1, -3, 2, -5, 6]], dtype=float)
b = np.array([15, -6, 20, -4, 7], dtype=float)"""
"""
A = np.array([[1, 2, -3, 4, -1, 1], 
              [-2, 3, 5, -1, 2, -1], 
              [4, -1, 2, 6, -3, 1],
              [-3, 5, -1, 2, 4, -1],
              [2, -4, 6, -5, 1, 3],
              [-5, 1, 4, -1, 2, -6]], dtype=float)
b = np.array([7, -2, 10, 3, -8, 5], dtype=float)
"""

# Resolución del sistema
sol = gauss_elimination(A, b)

# Imprimir la solución
print("Solución del sistema:")
print(sol)

