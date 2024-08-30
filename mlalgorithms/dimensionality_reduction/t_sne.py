import torch
from mlalgorithms.model import Model
from mlalgorithms.distances import squared_euclidian_pairwise_distances

from tqdm import tqdm

class tSNE(Model):
    """
    Implementation of t-SNE based on the paper "Visualizing Data using t-SNE" 
    (https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)
    
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

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor):
        # In this case Y is not used since t-SNE does not require any training

        nb_samples, _ = X.shape
        p_probs = self.p_joint_probs(X, self.perplexity)

        history = []
        y = torch.normal(mean=0.0, std=1e-4, size=(nb_samples, self.nb_dims))
        history.append(y); history.append(y)

        for i in tqdm(range(self.max_iter)):
            q_probs = self.q_joint_probs(history[-1])
            gradient = self.grad(p_probs, q_probs, history[-1])
            m_t = 0.5 if i < 250 else 0.8
            y = history[-1] - self.learning_rate * gradient + m_t * (history[-1] - history[-2])
            history.append(y)
        self.y = y
        return self

    def predict(self, X: torch.Tensor):
        self.fit(X, None)
        return self.y

    def p_conditional_probs(self, distances: torch.Tensor, stds: torch.Tensor):
        """
        Returns the conditional probabilities of each pair of samples
        using the given distances matrix and the stds

        distances   : shape [nb_samples, nb_samples]
        stds        : shape [nb_samples]
        """
        numerator = torch.exp(-distances / (2 * torch.square(stds.reshape((-1,1)))))
        numerator.fill_diagonal_(0.0)
        numerator += self.epsilon
        denominator = numerator.sum(axis=1).reshape([-1, 1])
        return numerator / denominator

    def compute_perplexity(self, conditional_probabilities: torch.Tensor) -> float:
        """
        Computes the perplexity of a matrix of conditional probabilities
        
        Returns :
        Perplexity value
        """
        return 2 ** (-torch.sum(conditional_probabilities * torch.log2(conditional_probabilities), axis=1))

    def find_stds(self, distances, perplexity):
        """
        Finds the stds value for each corresponding sample given perplexities
        distances   : shape [nb_samples, nb_samples]
        perplexity  : perplexity value goal

        Returns :
        [nb_samples] array containing the ideal std values of each sample
        """
        nb_samples = distances.shape[0]

        stds = torch.zeros((nb_samples))
        for i in range(nb_samples):
            current_sample_distances = distances[i:i+1, :]

            # Find best perplexity value using binary search
            guess = self.initial_guess
            upper_bound = self.upper_bound_search
            lower_bound = self.lower_bound_search
            for _ in range(self.max_iter_search):
                perplexity_value = self.compute_perplexity(self.p_conditional_probs(current_sample_distances, torch.Tensor([guess])))
                if torch.abs(perplexity_value - perplexity) < self.epsilon:
                    break
                if perplexity_value > perplexity:
                    upper_bound = guess
                else:
                    lower_bound = guess
                guess = (upper_bound + lower_bound) / 2
            stds[i] = guess

        return stds
    
    def p_joint_probs(self, samples: torch.Tensor, perplexity):
        distances = squared_euclidian_pairwise_distances(samples)
        stds = self.find_stds(distances, perplexity)
        conditionals = self.p_conditional_probs(distances, stds)
        return (conditionals + conditionals.T) / (2 * distances.shape[0])

    def q_joint_probs(self, samples: torch.Tensor):
        """
        Computes the joint distribution used in the low dimensional representation between pairs of samples 
        using a t-student distribution
        """
        distances = squared_euclidian_pairwise_distances(samples)
        numerator = 1 / (1 + distances)
        numerator.fill_diagonal_(0.0)
        denominator = torch.sum(torch.sum(numerator))
        return numerator / denominator

    def grad(self, p_joint: torch.Tensor, q_joint: torch.Tensor, samples: torch.Tensor):
        """
        Computes the gradient of each sample w.r.t to the KL-divergence
        """
        prob_diffs = torch.unsqueeze(p_joint - q_joint, 2)
        sample_diffs = torch.unsqueeze(samples, 1) - torch.unsqueeze(samples, 0)
        dist_term = torch.unsqueeze(1 / (1 + squared_euclidian_pairwise_distances(samples)), 2)
        return 4 * torch.sum(prob_diffs * sample_diffs * dist_term, axis=1)

