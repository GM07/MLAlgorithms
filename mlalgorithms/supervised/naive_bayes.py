from abc import ABC, abstractmethod
from mlalgorithms.model import Model

import torch

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

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Fits the model to the data

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        Y   :   (nb_samples,)
        array containing the prediction for all samples
        """
        nb_samples, self.nb_features = X.shape
        self.labels, counts = torch.unique(Y, return_counts=True)

        self.nb_classes = len(self.labels)
        self.priors = counts / nb_samples

        self.means = torch.zeros((self.nb_classes, self.nb_features))
        self.vars = torch.zeros((self.nb_classes, self.nb_features))

        for c in range(self.nb_classes):
            # `x_class` contains the samples of the current class
            x_class = X[Y == self.labels[c]]

            # Average along the samples to get the mean of all features
            self.means[c] = x_class.mean(axis=0) 

            # Same thing is done for the variance
            self.vars[c] = x_class.var(axis=0) + self.var_smoothing

        return self

    def predict(self, X: torch.Tensor):
        """
        Predicts the classes of samples

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        """
        nb_samples = X.shape[0]
        
        # We will store the probabilities in a matrix for each sample
        # Note that these probabilities will be computed in log space
        # for numerical stability
        probabilities = torch.zeros((nb_samples, self.nb_classes))

        probabilities += torch.log(self.priors)

        for c in range(self.nb_classes):
            probabilities[:, c] += GaussianNaiveBayes.pdf(X, self.means[c], self.vars[c])

        returned_value = self.labels[torch.argmax(probabilities, axis=1)]
        return returned_value

    @staticmethod
    def pdf(X: torch.Tensor, mean, var):
        """
        Computes the gaussian probabilty given an input X, a mean and a variance in the log space
        """
        return -0.5 * torch.sum(torch.log(2 * torch.pi * torch.sqrt(var)) + (X - mean) ** 2 / torch.sqrt(var), axis=1)


class BernoulliNaiveBayes(Model):

    def __init__(self) -> None:
        super().__init__()
        self.priors = None
        self.conditional_probs = None
        self.nb_classes = None

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Fits the model to the data

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        Y   :   (nb_samples,)
        array containing the prediction for all samples
        """
        nb_samples, self.nb_features = X.shape
        self.labels, counts = torch.unique(Y, return_counts=True)

        self.nb_classes = len(self.labels)
        self.priors = counts / nb_samples

        self.conditional_probs = torch.zeros((self.nb_classes, self.nb_features), dtype=torch.float32)

        for c in range(self.nb_classes):
            # `x_class` contains the samples of the current class
            x_class = X[Y == self.labels[c]]

            # Average along the samples of the given class to get the mean of all features
            self.conditional_probs[c] = (1 + x_class.sum(axis=0)) / (counts[c] + 2)

        return self

    def predict(self, X: torch.Tensor):
        """
        Predicts the classes of samples

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        """
        probabilities = torch.log(self.priors) + X @ torch.log(self.conditional_probs.T)
        return self.labels[torch.argmax(probabilities, axis=1)]


class MultinomialNaiveBayes(Model):

    def __init__(self, alpha_smoothing = 1.0) -> None:
        super().__init__()
        self.priors = None
        self.nb_classes = None
        self.alpha_smoothing = alpha_smoothing 

    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Fits the model to the data

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        Y   :   (nb_samples,)
        array containing the prediction for all samples
        """
        nb_samples, self.nb_features = X.shape
        self.labels, self.counts = torch.unique(Y, return_counts=True)

        self.nb_classes = len(self.labels)
        self.log_priors = torch.log(self.counts / nb_samples)

        self.feature_count = torch.zeros((self.nb_classes, self.nb_features))

        for c in range(self.nb_classes):
            # `x_class` contains the samples of the current class
            self.feature_count[c] = X[Y == self.labels[c]].sum(dim=0)

        smoothed_fc = self.feature_count + self.alpha_smoothing
        smoothed_cc = smoothed_fc.sum(dim=1).unsqueeze(1)
        
        self.log_conditional_probs = torch.log(smoothed_fc) - torch.log(smoothed_cc)
        
        return self

    def predict(self, X: torch.Tensor):
        """
        Predicts the classes of samples

        X   :   (nb_samples, nb_features)
        array containing the features of all samples
        
        """
        probabilities = self.log_priors + X @ self.log_conditional_probs.T
        return self.labels[torch.argmax(probabilities, axis=1)]

