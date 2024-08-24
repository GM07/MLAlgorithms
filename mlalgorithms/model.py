from abc import ABC, abstractmethod

import torch.nn as nn
import torch

class Model(nn.Module):

    @abstractmethod
    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Fits the model to the data

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        Y   :   (nb_samples,)
        array containing the prediction for all samples
        """
        return self

    @abstractmethod
    def predict(self, X: torch.Tensor):
        """
        Predicts the classes of samples

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        """
        pass
