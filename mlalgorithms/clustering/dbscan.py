from mlalgorithms.model import Model
from mlalgorithms.distances import euclidian

import torch
import random

class DBSCAN(Model):

    def __init__(self, min_samples, epsilon) -> None:
        self.min_samples = min_samples
        self.epsilon = epsilon

    def fit(self, X: torch.Tensor, Y: torch.Tensor):

        return self


    def predict(self, X: torch.Tensor):
        
        return self.fit(X, None)
