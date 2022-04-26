"""Ejercicio 1."""
import numpy as np

def normalizacion(x):
    """
    Funcion normalizacion().
    
    Calcula el z-score de los datos dados en el argumento x de manera vectorizada.
    """
    xmean_by_cols = np.mean(x, axis=0)
    xstd_by_cols = np.std(x, axis=0)
    z = (x-xmean_by_cols)/xstd_by_cols
    return z