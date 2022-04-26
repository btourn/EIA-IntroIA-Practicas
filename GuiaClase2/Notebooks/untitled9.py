import scipy.spatial.distance
import numpy as np

X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]) 
C = np.array([[1, 0, 0], [0, 1, 1]])
dist_scipy = scipy.spatial.distance.cdist(X, C)
print(dist_scipy)
#print(np.sqrt(np.sum((X[0,]-C[0,])**2)))
#print(np.sqrt(np.sum((X[0,]-C[1,])**2)))

def distancia(x,y):
    """
    Funcion distancia().
    
    Calcula la distancia de un punto X a un punto Y utilizando operaciones
    vectorizadas y broadcasting.
    """
    ny, dim = np.shape(y)
    XX = np.tile(x,(ny, 1, 1)) #Repite el array X "ny" veces
    CC = np.reshape(y, (ny, 1, dim)) #Reordena el array y para poder broadcastear con XX
    dist = np.sqrt(np.sum((XX-CC)**2, 2)) #Calculo vectorizado de la distancia
    idx = np.argmin(dist, axis=0) #Devuele los indices que corresponden a la distancia mas cercana
    return dist, idx


def k_means(x, n):
    """
    Funcion k_means().
    
    Implementacion propia del método k-means.
    """
    nx, dim = np.shape(x)
    rnd_idx = np.random.choice(np.arange(nx), size=n, replace=False)
    c0 = x[rnd_idx,]
    cent = np.copy(c0)
    max_iter = 20
    for i in range(max_iter):
        dist, idx = distancia(x, cent)
        cent = np.zeros([n, dim])
        for j in range(n):
            cent[j,] = np.mean(x[idx==j,], axis=0)
        
    return c0, cent, idx


x = np.random.rand(20,2)
n = 3
c0, cent, idx = k_means(x, n)
print(c0)
print(cent)
print(idx)
#print(f"Posicion inicial centroides: {c0:.4f}")
#print(f"Posicion final centroides: {cent:.4f}")
#print(f"Pertenencia de los indices del dataset a cada clase: {idx:d}")

## Prueba utilizando Kmeans de sklearn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=n, init=c0).fit(x)
kmeans.cluster_centers_
