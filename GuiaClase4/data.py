import numpy as np
import matplotlib.pyplot as plt

class Data(object):

    def __init__(self, n):
        self.X, self.y = self.build_dataset(n)
        
        
    @staticmethod
    def _analytic_function(X):
        return (np.abs(X**2-5)) + np.sin(2*X)*np.cos(X**3)
        

    def build_dataset(self, n):
        sigma_epsilon = 1
        x_max = 4
        x = x_max * (2 * np.random.rand(n) - 1)
        epsilon = sigma_epsilon * np.random.randn(n)
        y = self._analytic_function(x)
        y = y + epsilon
        return x, y

    def split(self, percentage, n_seed=0):
        X = self.X
        y = self.y
        
        np.random.seed(n_seed)
        permuted_idxs = np.random.permutation(X.shape[0])

        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]

        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test
    
    def plot_original(self):
        plt.figure(figsize=(12, 6))
        x_range = np.linspace(-4, 4, 1000)
        plt.scatter(self.X, self.y)
        plt.plot(x_range, self._analytic_function(x_range), 'r', linewidth=3.0)
        plt.xlabel('x', size=12)
        plt.ylabel('y', size=12)
        plt.xticks(np.arange(-4-1, 4 + 1))
        plt.show()