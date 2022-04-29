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
        #self.train = train
        #data = self.validation
        #data = self.test
        return train, validation, test



    def removeNaNs(self, X):
        ncols = len(X[0])
        nrows = len(X)
        nans = np.array([], dtype="int64")
        nans_in_cols = []
        for i in range(ncols):
            icol = 'col' + str(i+1)
            aux = np.where(np.isnan(X[icol]))
            if aux[0].size != 0:
                nans_in_cols.append(icol)
            nans = np.append(nans, aux[0], axis=0)
        
        X = X[[name for name in X.dtype.names if name not in nans_in_cols]]
        nans = np.unique(nans)
        X = np.delete(X, nans)
        
        # Cambio de array estructurado a 2d array para poder pasar X a PCA de scikit-learn
        for i, col_name in enumerate(X.dtype.names):
            if i==0:
                XX = X[col_name]
            else:
                XX = np.c_[XX, X[col_name]]
        
        return XX
    
    
    def replaceNaNs(self, X):
        ncols = len(X[0])
        nrows = len(X)
        for i in range(ncols):
            icol = 'col' + str(i+1)
            aux = np.where(np.isnan(X[icol]))
            if aux[0].size != 0:
                col_mean = np.nanmean(X[icol])
                X[icol][aux] = col_mean
        
        # Cambio de array estructurado a 2d array para poder pasar X a PCA de scikit-learn
        for i, col_name in enumerate(X.dtype.names):
            if i==0:
                XX = X[col_name]
            else:
                XX = np.c_[XX, X[col_name]]
        
        return XX

