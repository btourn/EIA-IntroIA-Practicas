"""Ejercicio 3."""
import numpy as np

def replaceNaNs(x):
    """
    Funcion replaceNaNs().
    
    Reemplaza los NaNs encontrados en el dataset por la media de los demás valores
    de la misma columna.
    El argumento de entrada 'x' debe ser un numpy array de nxm.
    """
    rows, cols = np.where(np.isnan(x))
    mean_wo_nans = np.nanmean(x[:, cols], axis=0)
    x[rows, cols] = mean_wo_nans
    return x

# x = np.array([[1, np.nan, 3], [4, 5, np.nan], [7, 8, 9], [10, 11, 12], [0, 1, 1]])
# replaceNaNs(x)