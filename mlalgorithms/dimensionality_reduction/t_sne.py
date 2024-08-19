from numpy.typing import NDArray
from mlalgorithms.model import Model

import numpy as np
from tqdm import tqdm

class tSNE(Model):
    """
    Implementation derived from https://towardsdatascience.com/understanding-t-sne-by-implementing-2baf3a987ab3
    """

    def __init__(
        self, 
        nb_dims: int = None, 
        epsilon = 1e-4, 
        max_iter: int = 1000, 
        learning_rate: float = 200.0,
        perplexity: float = 40,
    ) -> None:
        self.nb_dims = nb_dims
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.perplexity = perplexity

        self.lower_bound_search = 1e-10
        self.upper_bound_search = 1e5
        self.max_iter_search = 1000
        self.initial_guess = (self.lower_bound_search + self.upper_bound_search) / 2

        super().__init__()

    def fit(self, X: NDArray, Y: NDArray):
        # In this case Y is not used since t-SNE does not require any training

        nb_samples, _ = X.shape
        p_probs = self.p_joint_probs(X, self.perplexity)

        history = []
        y = np.random.normal(loc=0.0, scale=1e-4, size=(nb_samples, self.nb_dims))
        history.append(y); history.append(y)

        for i in tqdm(range(self.max_iter)):
            q_probs = self.q_joint_probs(history[-1])
            gradient = self.grad(p_probs, q_probs, history[-1])
            m_t = 0.5 if i < 250 else 0.8
            y = history[-1] - self.learning_rate * gradient + m_t * (history[-1] - history[-2])
            history.append(y)
        return y

    def predict(self, X: NDArray):
        return self.fit(X, None)

    def euclidian_pairwise_distances(self, X: NDArray):
        """
        Computes pairwise euclidian distance between every pair of points 
        in the sample matrix X
        """
        return np.sum((X[None, :] - X[:, None]) ** 2, axis=2)

    def p_conditional_probs(self, distances: NDArray, stds: NDArray):
        """
        Returns the conditional probabilities of each pair of samples
        using the given distances matrix and the stds

        distances   : shape [nb_samples, nb_samples]
        stds        : shape [nb_samples]
        """
        numerator = np.exp(-distances / (2 * np.square(stds.reshape((-1,1)))))
        np.fill_diagonal(numerator, 0.0)
        numerator += self.epsilon
        denominator = numerator.sum(axis=1).reshape([-1, 1])
        return numerator / denominator

    def compute_perplexity(self, conditional_probabilities: NDArray) -> float:
        """
        Computes the perplexity of a matrix of conditional probabilities
        
        Returns :
        Perplexity value
        """
        return 2 ** (-np.sum(conditional_probabilities * np.log2(conditional_probabilities), axis=1))

    def find_stds(self, distances, perplexity):
        """
        Finds the stds value for each corresponding sample given perplexities
        distances   : shape [nb_samples, nb_samples]
        perplexity  : perplexity value goal

        Returns :
        [nb_samples] array containing the ideal std values of each sample
        """
        nb_samples = distances.shape[0]

        stds = np.zeros((nb_samples))
        for i in range(nb_samples):
            current_sample_distances = distances[i:i+1, :]

            # Find best perplexity value using binary search
            guess = self.initial_guess
            upper_bound = self.upper_bound_search
            lower_bound = self.lower_bound_search
            for _ in range(self.max_iter_search):
                perplexity_value = self.compute_perplexity(self.p_conditional_probs(current_sample_distances, np.array([guess])))
                if np.abs(perplexity_value - perplexity) < self.epsilon:
                    break
                if perplexity_value > perplexity:
                    upper_bound = guess
                else:
                    lower_bound = guess
                guess = (upper_bound + lower_bound) / 2
            stds[i] = guess

        return stds
    
    def p_joint_probs(self, samples: NDArray, perplexity):
        distances = self.euclidian_pairwise_distances(samples)
        stds = self.find_stds(distances, perplexity)
        conditionals = self.p_conditional_probs(distances, stds)
        return (conditionals + conditionals.T) / (2 * distances.shape[0])

    def q_joint_probs(self, samples: NDArray):
        """
        Computes the joint distribution used in the low dimensional representation between pairs of samples 
        using a t-student distribution
        """
        distances = self.euclidian_pairwise_distances(samples)
        numerator = 1 / (1 + distances)
        np.fill_diagonal(numerator, 0.0)
        denominator = np.sum(np.sum(numerator))
        return numerator / denominator

    def grad(self, p_joint: NDArray, q_joint: NDArray, samples: NDArray):
        """
        Computes the gradient of each sample w.r.t to the KL-divergence
        """
        prob_diffs = np.expand_dims(p_joint - q_joint, 2)
        sample_diffs = np.expand_dims(samples, 1) - np.expand_dims(samples, 0)
        dist_term = np.expand_dims(1 / (1 + self.euclidian_pairwise_distances(samples)), 2)
        return 4 * np.sum(prob_diffs * sample_diffs * dist_term, axis=1)

