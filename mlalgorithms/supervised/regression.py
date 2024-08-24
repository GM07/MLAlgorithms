from mlalgorithms.model import Model

import torch

class LinearRegression(Model):

    """
    
    Basic linear regression model solving the equation Y = Xa + b
    using the equation : a = (X^T * X)^-1 * X^T * y. The bias is 
    added within the matrix a

    """

    def __init__(self) -> None:
        super().__init__()

        self._weights = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Fits the linear model to the data

        X : tensor of shape (nb_samples, nb_features)

        Y : tensor of shape (nb_samples)
        """
        X = self.add_bias(X)
        moore_penrose_inv: torch.Tensor = torch.linalg.inv(X.T.matmul(X) + self.get_regularization_coef(X.shape[-1]))
        self._weights = moore_penrose_inv.matmul(X.T).matmul(Y)
        return self

    def predict(self, X: torch.Tensor):
        return self.add_bias(X) @ self._weights

    def add_bias(self, X: torch.Tensor):
        return torch.concatenate([X, torch.ones((X.shape[0], 1))], axis=1)

    def get_regularization_coef(self, nb_features: int):
        return torch.zeros((nb_features, nb_features))


class RidgeRegression(LinearRegression):

    def __init__(self, regularization_coef: float) -> None:
        super().__init__()
        self.regularization_coef = regularization_coef

    def get_regularization_coef(self, nb_features):
        return self.regularization_coef * torch.eye(nb_features)

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

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        X_bias = self.add_bias(torch.tensor(X))
        Y = torch.tensor(Y)
        if len(Y.shape) == 1:
            Y = torch.unsqueeze(torch.tensor(Y), -1)
        _, nb_features = X_bias.shape
        self._weights = torch.normal(0, 1, (nb_features, 1))

        for _ in range(self.nb_epochs):
            grad = self.grad(X_bias, Y)
            self._weights -= self.learning_rate * grad

    def predict(self, X: torch.Tensor):
        return super().predict(torch.tensor(X))

    def grad(self, X: torch.Tensor, Y: torch.Tensor):
        error = X.matmul(self._weights) - Y
        grad = (1 / Y.shape[0]) * X.T.matmul(error) + 2 * self.regularization_coef * self._weights
        penalty = torch.zeros(size=self._weights.shape)
        for i in range(X.shape[1]):
            penalty[i] = 1 if self._weights[i] > 0 else -1
        return grad + self.regularization_coef * penalty

class LogisticRegression(LinearRegression):
    """Linear regression with a sigmoid activation function on top"""

    def __init__(
        self,
        learning_rate = 0.1, 
        nb_epochs = 50,
        threshold = 0.5,
        epsilon = 1e-6
    ) -> None:
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        self.threshold = threshold
        self.epsilon = epsilon
        super().__init__()

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        X_bias = self.add_bias(torch.Tensor(X))
        Y = torch.Tensor(Y)
        if len(Y.shape) == 1:
            Y = torch.unsqueeze(torch.Tensor(Y), -1)

        self.labels = torch.unique(Y).tolist()
        assert len(self.labels) == 2, "This classifier only supports two classes"
        assert set(self.labels) == {0., 1.}, f"The labels must be 0 and 1, but are {set(self.labels)}"

        _, nb_features = X_bias.shape
        self._weights = torch.normal(0, 1, (nb_features, 1))

        for _ in range(self.nb_epochs):
            grad = self.grad(X_bias, Y)
            if torch.linalg.norm(grad) < self.epsilon:
                break

            self._weights -= self.learning_rate * grad

    def predict(self, X: torch.Tensor):
        probas = self.sigmoid(super().predict(torch.Tensor(X)))
        return torch.where(probas > 0.5, 1, 0)

    def grad(self, X: torch.Tensor, Y: torch.Tensor):
        predictions = self.sigmoid(X.matmul(self._weights))
        grad = (1 / Y.shape[0]) * X.T.matmul(predictions - Y)
        return grad

    def sigmoid(self, X: torch.Tensor):
        return 1 / (1 + torch.exp(-X))

