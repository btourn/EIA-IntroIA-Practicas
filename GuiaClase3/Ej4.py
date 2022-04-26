"""Ejercicio 4."""
import numpy as np

def define_subsets(x, pct_train, pct_val, pct_test):
    """
    Funcion define_subsets().
    
    Realiza la partición del dataset en conjuntos de train, validation y test.
    """
    nx = np.size(x, axis=0)
    idx0 = np.arange(nx)
    idx = np.random.permutation(idx0)
    percentages = np.array([pct_train, pct_val, pct_test])*nx
    n_by_set = np.ceil(percentages).astype(int)
    [n_train, n_val, n_test] = n_by_set 
    n_to_remove = np.sum(n_by_set)-nx
    n_train = n_train-n_to_remove
    
    idx_train = idx[np.arange(n_train)]
    train = x[idx_train, ]
    idx_val = idx[np.arange(n_train, n_val+n_train)]
    validation = x[idx_val, ]
    idx_test = idx[np.arange(n_val+n_train, n_test+n_val+n_train)]
    test = x[idx_test, ]
    return train, validation, test

#x = np.random.rand(87,4)
#define_subsets(x, 0.7, 0.2, 0.1)