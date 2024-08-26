from mlalgorithms.model import Model

import torch

class PCA(Model):

    def __init__(self, nb_dims: int = None) -> None:
        self.nb_dims = nb_dims
        super().__init__()

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        # In this case Y is not used since PCA does not require any training

        # Data needs to be centered before applying PCA
        self.mean = torch.unsqueeze(X.mean(axis=0), axis=0)
        X_centered = X - self.mean

        self.U, self.S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
        self.U, Vt = self.svd_flip_sign(self.U, Vh)
        self.V = Vt[:self.nb_dims]
        return self

    def predict(self, X: torch.Tensor):
        self.fit(X, None)
        return (X - self.mean) @ self.V.T

    @staticmethod
    def svd_flip_sign(u, v):
        max_abs_indices = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_indices, range(u.shape[1])])
        u *= signs
        v *= signs[:, None]
        return u, v
