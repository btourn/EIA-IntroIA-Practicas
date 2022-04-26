"""Metricas definidas a partir de clases."""

import numpy as np

class Metricas(object):
    """Clase 'Metricas' base."""
    
    def __call__(self, target, prediction):
        return NotImplemented
    
    
class MSE(Metricas):
    """Clase 'MSE' que hereda de 'Metricas'."""
    def __init__(self):
        Metricas.__init__(self)

    def __call__(self, target, prediction):
        return np.dot(target-prediction, target-prediction)/len(target)
        

