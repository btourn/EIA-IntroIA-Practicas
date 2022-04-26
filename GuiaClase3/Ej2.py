"""Ejercicio 2."""
import numpy as np

def removeNaNs(x):
    """
    Funcion removeNaNs().
    
    Remueve las filas y columnas donde encuentre un valor NaN.
    El argumento de entrada 'x' debe ser un numpy array de nxm.
    """
    rows, cols = np.where(np.isnan(x))
    x = np.delete(x, rows, axis=0)
    x = np.delete(x, cols, axis=1)
    return x