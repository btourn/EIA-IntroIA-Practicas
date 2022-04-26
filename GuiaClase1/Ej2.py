"""Sorting."""

import numpy as np
from Ej1 import norma_p

def ordena(A, nA):
    """
    Funcion ordena.
    
    A: matriz de nxm. nA: vector que contiene las norma-2 de las filas de A.
    La función ordena al vector nA de menor a mayor y se obtienen los indices
    correspondientes. Con esos indices se reordena la matriz A. Devuelve la
    matriz A ordenada y el vector de normas nA ordenado.
    """
    normas_ord = np.sort(nA)
    idx = np.argsort(nA)
    A_ord = A[idx,]
    return A_ord, normas_ord


# X = np.random.rand(10,5)
# norma_X = norma_p(X, 2)
# X_ord, normas_ord = ordena(X, norma_X)

# print(normas_ord.T)
# print(X_ord)
