from mlalgorithms.model import Model

import numpy as np

class LinearRegression(Model):

    """
    
    Basic linear regression model solving the equation Y = Xa + b
    using the equation : a = (X^T * X)^-1 * X^T * y. The bias is 
    added within the matrix a

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
        moore_penrose_inv = np.linalg.inv((X.T.dot(X) + self.get_regularization_coef(X.shape[1])))
        self._weights = moore_penrose_inv.dot(X.T).dot(Y)
        return self

    def predict(self, X):
        return self.add_bias(X) @ self._weights

    def add_bias(self, X):
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    def get_regularization_coef(self, nb_features: int):
        return np.zeros((nb_features, nb_features))


class RidgeRegression(LinearRegression):

    def __init__(self, regularization_coef: float) -> None:
        super().__init__()
        self.regularization_coef = regularization_coef

    def get_regularization_coef(self, nb_features):
        return self.regularization_coef * np.identity(nb_features)
