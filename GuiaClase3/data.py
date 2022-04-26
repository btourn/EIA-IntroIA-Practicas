"""Clase Data."""

import numpy as np

class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    @staticmethod
    def _build_dataset(path):
        structure = [('col1', np.float), ('col2', np.float), ('col3', np.float),
                     ('col4', np.float), ('col5', np.float), ('col6', np.float),
                     ('col7', np.float)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(';')[0]), float(line.split(';')[1]), 
                        float(line.split(';')[2]), float(line.split(';')[3]),
                        float(line.split(';')[4]), float(line.split(';')[5]), 
                        float(line.split(';')[6]))
                        for i, line in enumerate(data_csv))
            data = np.fromiter(data_gen, structure)

        return data

    def split(self, pct_train, pct_val, pct_test):  # 0.8
        X = self.dataset #Paso todo el dataset completo

        nx = np.size(X, axis=0)
        idx0 = np.arange(nx)
        idx = np.random.permutation(idx0)
        percentages = np.array([pct_train, pct_val, pct_test])*nx
        n_by_set = np.ceil(percentages).astype(int)
        [n_train, n_val, n_test] = n_by_set 
        n_to_remove = np.sum(n_by_set)-nx
        n_train = n_train-n_to_remove
    
        idx_train = idx[np.arange(n_train)]
        train = X[idx_train, ]
        idx_val = idx[np.arange(n_train, n_val+n_train)]
        validation = X[idx_val, ]
        idx_test = idx[np.arange(n_val+n_train, n_test+n_val+n_train)]
        test = X[idx_test, ]
        return train, validation, test

    
    def removeNaNs(self):
        X = self.dataset
        ncols = len(X[0])
        for i in range(ncols):
            icol = 'col' + str(i+1)
            asd = np.where(np.isnan(X[icol]))
            
        
        
        rows, cols = np.where(np.isnan(x))
        x = np.delete(x, rows, axis=0)
        x = np.delete(x, cols, axis=1)
        return x
    
    def replaceNaNs(self):
        pass

