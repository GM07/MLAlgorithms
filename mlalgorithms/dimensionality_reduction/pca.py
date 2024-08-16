from numpy.typing import NDArray
from mlalgorithms.model import Model

import numpy as np

class PCA(Model):

    def __init__(self, nb_components: int = None) -> None:
        self.nb_components = nb_components
        super().__init__()

    def fit(self, X: NDArray, Y: NDArray):
        # In this case Y is not used since PCA does not require any training

        # Data needs to be centered before applying PCA
        X_centered = X - np.expand_dims(X.mean(axis=0), axis=0)
        
        self.u, self.sigma, self.v = np.linalg.svd(X_centered)
        return self.u, self.sigma, self.v

    def predict(self, X: NDArray):
        self.fit(X, None)
        return np.expand_dims(self.sigma, axis=-1) * self.u[:self.nb_components]
