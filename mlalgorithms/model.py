from abc import ABC, abstractmethod
import numpy as np

class Model(ABC):

    @abstractmethod
    def fit(self, x, y):
        """
        x   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        y   :   (nb_samples,)
        array containing the prediction for all features
        """
        pass

    @abstractmethod
    def predict(self, x):
        pass
