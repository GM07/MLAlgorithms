from mlalgorithms.model import Model
from mlalgorithms.distances import euclidian

import torch
import random

class KMeans(Model):

    def __init__(self, nb_clusters, nb_iterations) -> None:
        super().__init__()
        self.nb_clusters = nb_clusters
        self.nb_iterations = nb_iterations

    @torch.no_grad()
    def fit(self, X: torch.Tensor, Y: torch.Tensor = None):
        """
        Fits the centroids to the clusters present in the data

        X : numpy array of shape (nb_samples, nb_features)
        """
        nb_samples, _ = X.shape

        # Choose `nb_clusters` random points
        indices = random.sample(range(nb_samples), k=self.nb_clusters)

        # Assign the centroids to these points
        centroids = X[indices, :].clone()
        
        # Assign the points to the clusters
        assigments = torch.zeros(nb_samples, dtype=torch.int64)
        assigments[indices] = torch.tensor(range(self.nb_clusters))

        for i in range(self.nb_iterations):
            # Compute new mean of centroids
            for cluster in range(self.nb_clusters):
                cluster_samples = torch.where(assigments == cluster)
                if len(cluster_samples) > 0:
                    centroids[cluster] = X[cluster_samples].mean(axis=0)

            old_assigments = assigments.clone()

            # Re-assign the data points to the clusters   
            for i, sample in enumerate(X):

                # Find closest centroid
                current_cluster = assigments[i]
                for j, centroid in enumerate(centroids):
                    dist_current_centroid = euclidian(centroids[current_cluster], sample)
                    dist_centroid = euclidian(centroid, sample)

                    if dist_centroid < dist_current_centroid:
                        assigments[i] = j

            if torch.all(old_assigments == assigments):
                break

        self.clusters_centroids = centroids
        self.labels = assigments
        return self

    def predict(self, X: torch.Tensor):
        """
        Predicts to which clusters a set of samples belongs to

        x : numpy array of shape (nb_samples, nb_features)
        """
        assert self.clusters_centroids is not None, "The fit() method must be called first"

        nb_samples, _ = X.shape

        closest_clusters = torch.zeros(nb_samples, dtype=int)
        for sample_index, sample in enumerate(X):
            current_cluster = closest_clusters[sample_index]
            for cluster in range(1, self.clusters_centroids.shape[0]):
                current_dist = euclidian(self.clusters_centroids[current_cluster], sample)
                cluster_dist = euclidian(self.clusters_centroids[cluster], sample)
                if cluster_dist < current_dist:
                    closest_clusters[sample_index] = cluster

        return closest_clusters
