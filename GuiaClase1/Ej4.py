"""Precision, recall y accuracy."""

import numpy as np

def metricas_basicas(x, y):
    """
    Funcion metricas_basicas.
    
    x: lista de valores verdaderos.
    y: lista de valores predichos.
    
    La funcion calcula las variables true positive (TP), true negative (TN),
    false negative (FN) y false positive (FP). Luego, calcula las métricas:
    
        Precision: P = TP/(TP+FP)
        Recall:    R = TP/(TP+FN)
        Accuracy:  A = (TP+TN)/(TP+TN+FN+FP)
    """
    Tr = np.array(x)
    Pr = np.array(y)
    TP = np.size(np.where(Pr+Tr==0)) #sum(Tr & Pr)
    TN = np.size(np.where(Pr+Tr==2))
    FN = np.size(np.intersect1d(np.where(Pr+Tr==1),np.where(Tr)))
    FP = np.size(np.intersect1d(np.where(Pr+Tr==1),np.where(Pr)))
    
    P = TP/(TP+FP)
    R = TP/(TP+FN)
    A = (TP+TN)/(TP+TN+FN+FP)
    
    return P, R, A


# truth = [1, 1, 0, 1, 1, 1, 0, 0, 0, 1]
# prediction = [1, 1, 1, 1, 0, 0, 1, 1, 0, 0]
# precision, recall, accuracy = metricas_basicas(truth, prediction)
