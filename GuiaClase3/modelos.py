"""Modelos definidos a partir de clases."""

import numpy as np

class ModeloBase(object):

    def __init__(self):
        self.model = None

    def fit(self, X, Y):
        return NotImplemented

    def predict(self, X):
        return NotImplemented


class RegresionLineal(ModeloBase):

    def fit(self, X, y):
        X_expanded = np.vstack((X.T, np.ones(len(X)))).T
        W = np.linalg.inv(X_expanded.T.dot(X_expanded)).dot(X_expanded.T).dot(y)
        self.model = W

    def predict(self, X):
        X_expanded = np.vstack((X.T, np.ones(len(X)))).T
        return X_expanded.dot(self.model)

