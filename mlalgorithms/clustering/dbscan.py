from collections import deque
from mlalgorithms.model import Model
from mlalgorithms.distances import euclidian_pairwise_distances

import torch

class DBSCAN(Model):

    """
    Based on the paper "A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise" (https://cdn.aaai.org/KDD/1996/KDD96-037.pdf)
    """

    def __init__(self, min_samples, epsilon) -> None:
        """
        min_samples : Minimum number of points within an epsilon distance of a sample for that sample to be considered a core sample
        epsilon     : Maximum distance to consider when evaluating core samples
        """
        self.min_samples = min_samples
        self.epsilon = epsilon

    def fit(self, X: torch.Tensor, Y: torch.Tensor):

        nb_samples, _ = X.shape
        distances: torch.Tensor = euclidian_pairwise_distances(X)        
        neighbors = torch.where(distances <= self.epsilon, 1, 0)
        neighbors_count = neighbors.sum(dim=0)

        self.core_samples = torch.where(neighbors_count >= self.min_samples, True, False)
        self.labels = torch.zeros((nb_samples), dtype=torch.int64) - 1

        current_cluster = 0
        for i in range(nb_samples):
            if self.labels[i] != -1 or not self.core_samples[i]:
                continue

            # We found a core sample. We now need to expand the cluster using core samples
            self.expand_cluster(i, neighbors, current_cluster)
            current_cluster += 1
        return self

    def predict(self, X: torch.Tensor):
        self.fit(X, torch.Tensor([]))
        return self.labels

    def expand_cluster(self, starting_core_sample: int, neighbors: torch.Tensor, cluster_to_assign: int):
        """
        starting_core_sample    :   Index of sample which is a core sample that will be used as a start to find all neighbors
        neighbors               :   2d-Tensor containing for each sample i, if the sample j is considered to be its neighbor
        cluster_to_assign       :   To which cluster to assign the core samples/neighbors of the current core sample
        """
        queue = deque([starting_core_sample])
        while len(queue) > 0:
            current_sample = queue.popleft()
            if self.labels[current_sample] != -1:
                continue

            self.labels[current_sample] = cluster_to_assign

            neighbor_samples = neighbors[current_sample] > 0 & self.core_samples[current_sample]
            next_point_indices = neighbor_samples.nonzero()

            for next_point_index in next_point_indices:
                if self.labels[next_point_index] == -1:
                    queue.append(int(next_point_index.item()))
