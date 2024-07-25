from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):

    @abstractmethod
    def fit(self, X, Y):
        """
        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        Y   :   (nb_samples,)
        array containing the prediction for all features
        """
        pass

    @abstractmethod
    def predict(self, X):
        pass
