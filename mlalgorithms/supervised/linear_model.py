from mlalgorithms.model import Model

import numpy as np

class LinearRegression(Model):

    """
    
    Basic linear regression model solving the equation

    Y = Xa + b

    using the equation : a = (X^T * X)^-1 * X^T y

    """

    def __init__(self) -> None:
        super().__init__()

        self._weights = None

    def fit(self, X, Y):
        """
        Fits the linear model to the data

        X : numpy array of shape (nb_samples, nb_features)

        Y : numpy array of shape (nb_samples)
        """
        X = self.add_bias(X)
        self._weights = np.linalg.inv((X.T.dot(X))).dot(X.T).dot(Y)

    def predict(self, X):
        return self.add_bias(X) @ self._weights

    def add_bias(self, X):
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
