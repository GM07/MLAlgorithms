from abc import ABC, abstractmethod

import torch.nn as nn
import torch

class Model:

    @abstractmethod
    def fit(self, X: torch.Tensor, Y: torch.Tensor) -> 'Model':
        """
        Fits the model to the data

        X   :   (nb_samples, nb_features)
        tensor containing the features of all samples
        
        Y   :   (nb_samples,)
        tensor containing the prediction for all samples
        """
        return self

    @abstractmethod
    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """
        Predicts the classes of samples

        X   :   (nb_samples, nb_features)
        tensor containing the features of all samples
        
        """
        pass
