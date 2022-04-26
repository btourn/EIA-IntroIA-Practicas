"""Ejercicio 8: implementacion basica de k-means."""

import numpy as np
from Ej6 import distancia

def k_means(x, n):
    """
    Funcion k_means().
    
    asdasdas.
    """
    nx, dim = np.shape(x)
    rnd_idx = np.random.choice(np.arange(nx), size=n, replace=False)
    c0 = x[rnd_idx,]
    cent = np.copy(c0)
    N = 10
    for i in range(N):
        d, idx = distancia(x, cent)
        cent = np.zeros([n, dim])
        for j in range(n):
            cent[j,] = np.mean(x[idx==j,], axis=0)
        print(cent)
        
    return c0, cent, idx



# from sklearn.cluster import KMeans
# import matplotlib.pyplot as plt

# n_cl = 2 #Cantidad de clusters a crear
# x = np.random.randint(-10, high=10, size=(10, 2), dtype=int)
# #plt.scatter(x[:,0], x[:,1])
# #plt.show()

# c0, coord_centroids, idx = k_means(x, n_cl)

# kmeans = KMeans(n_clusters=n_cl, random_state=0, init=c0).fit(x)
# kmeans.cluster_centers_