"""Calculo vectorizado de normas l0, l1 y l2."""

import numpy as np

def norma_p(A, p):
    """
    Función norma_p.
    
    Calcula simultaneamente la norma p de los vectores fila x de la matriz A, segun
    la expresión
        
    ||x||^p = (sum_{i=1}^{n}|x_i|^p)^(1/p)
    
    donde x_i son las componentes de x. En caso de desear computar la norma infinito,
    la operacion a realizar es
    
    ||x||^{inf} = max_i(|x_i|).
    """
    if p=='inf':
        norma = np.array([np.max(np.abs(A),1)])
        return norma
        
    norma = np.array([np.power(np.sum(np.power(np.abs(A),p),1),1/p)])
    return norma
    
    
X = np.random.rand(10,5)

norma = norma_p(X, 'inf')
print(norma.T)