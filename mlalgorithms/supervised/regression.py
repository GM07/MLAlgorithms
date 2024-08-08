from numpy.typing import NDArray
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

    def fit(self, X: NDArray, Y: NDArray):
        """
        Fits the linear model to the data

        X : numpy array of shape (nb_samples, nb_features)

        Y : numpy array of shape (nb_samples)
        """
        X = self.add_bias(X)
        moore_penrose_inv = np.linalg.inv((X.T.dot(X) + self.get_regularization_coef(X.shape[1])))
        self._weights = moore_penrose_inv.dot(X.T).dot(Y)
        return self

    def predict(self, X: NDArray):
        return self.add_bias(X) @ self._weights

    def add_bias(self, X: NDArray):
        return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)

    def get_regularization_coef(self, nb_features: int):
        return np.zeros((nb_features, nb_features))


class RidgeRegression(LinearRegression):

    def __init__(self, regularization_coef: float) -> None:
        super().__init__()
        self.regularization_coef = regularization_coef

    def get_regularization_coef(self, nb_features):
        return self.regularization_coef * np.identity(nb_features)

class LassoRegression(LinearRegression):

    def __init__(
        self, 
        regularization_coef = 1.0, 
        learning_rate = 0.1, 
        nb_epochs = 50
    ) -> None:
        self.regularization_coef = regularization_coef
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs

    def fit(self, X: NDArray, Y: NDArray):
        X_bias = self.add_bias(np.array(X))
        Y = np.expand_dims(np.array(Y), -1)
        _, nb_features = X_bias.shape
        self._weights = np.random.normal(0, 1, (nb_features, 1))

        for _ in range(self.nb_epochs):
            grad = self.grad(X_bias, Y)
            self._weights -= self.learning_rate * grad

    def predict(self, X: NDArray):
        return super().predict(np.array(X))

    def grad(self, X: NDArray, Y: NDArray):
        error = X.dot(self._weights) - Y
        grad = (1 / Y.shape[0]) * X.T.dot(error) + 2 * self.regularization_coef * self._weights
        penalty = np.zeros(shape=self._weights.shape)
        for i in range(X.shape[1]):
            penalty[i] = 1 if self._weights[i] > 0 else -1
        return grad + self.regularization_coef * penalty
