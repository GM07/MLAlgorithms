from abc import ABC, abstractmethod
import numpy as np
from numpy.typing import NDArray

class Model(ABC):

    @abstractmethod
    def fit(self, X: NDArray, Y: NDArray):
        """
        Fits the model to the data

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        Y   :   (nb_samples,)
        array containing the prediction for all samples
        """
        return self

    @abstractmethod
    def predict(self, X: NDArray):
        """
        Predicts the classes of samples

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        """
        pass
