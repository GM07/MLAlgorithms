from abc import ABC, abstractmethod
from mlalgorithms.model import Model

import numpy as np
from numpy.typing import NDArray

class GaussianNaiveBayes(Model):

    def __init__(self, var_smoothing = 1e-9) -> None:
        super().__init__()
        self.priors = None
        self.means = None
        self.vars = None
        self.nb_classes = None

        # Used for smoothing the variance since we divide by the standard
        # deviation during predictions
        self.var_smoothing = var_smoothing 

    def fit(self, X: NDArray, Y: NDArray):
        """
        Fits the model to the data

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        Y   :   (nb_samples,)
        array containing the prediction for all samples
        """
        nb_samples, self.nb_features = X.shape
        self.labels, counts = np.unique(Y, return_counts=True)

        self.nb_classes = len(self.labels)
        self.priors = counts / nb_samples

        self.means = np.zeros((self.nb_classes, self.nb_features))
        self.vars = np.zeros((self.nb_classes, self.nb_features))

        for c in range(self.nb_classes):
            # `x_class` contains the samples of the current class
            x_class = X[Y == c]

            # Average along the samples to get the mean of all features
            self.means[c] = x_class.mean(axis=0) 

            # Same thing is done for the variance
            self.vars[c] = x_class.var(axis=0) + self.var_smoothing

        return self

    def predict(self, X: NDArray):
        """
        Predicts the classes of samples

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        """
        nb_samples = X.shape[0]
        
        # We will store the probabilities in a matrix for each sample
        # Note that these probabilities will be computed in log space
        # for numerical stability
        probabilities = np.zeros((nb_samples, self.nb_classes))

        probabilities += np.log(self.priors)

        for c in range(self.nb_classes):
            probabilities[:, c] += GaussianNaiveBayes.pdf(X, self.means[c], self.vars[c])

        return self.labels[np.argmax(probabilities, axis=1)]

    @staticmethod
    def pdf(X: NDArray, mean, var):
        """
        Computes the gaussian probabilty given an input X, a mean and a variance in the log space
        """
        return -0.5 * np.sum(np.log(2 * np.pi * np.sqrt(var)) + (X - mean) ** 2 / np.sqrt(var), axis=1)
