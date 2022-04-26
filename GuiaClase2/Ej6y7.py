"""Ejercicio 6 y 7: Calculo de distancias."""

import numpy as np

def distancia(x,y):
    """
    Funcion distancia().
    
    Calcula la distancia de un punto X a un punto Y utilizando operaciones
    vectorizadas y broadcasting.
    """
    import numpy as np
    ny, dim = np.shape(y)
    xx = np.tile(x,(ny, 1, 1)) #Repite el array x "ny" veces
    yy = np.reshape(y, (ny, 1, dim)) #Reordena el array y para poder broadcastear con xx
    dist = np.sqrt(np.sum((xx-yy)**2, 2)) #Calculo vectorizado de la distancia
    idx = np.argmin(dist, axis=0) #Devuele los indices que corresponden a la distancia mas cercana
    return dist, idx


#import numpy as np
#x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [0, 0, 0]])
#c = np.array([[1, 0, 0], [0, 1, 1]])
#distancia, idx_mas_cercano = distancia(x,c)