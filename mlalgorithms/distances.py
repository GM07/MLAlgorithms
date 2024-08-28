import torch


def euclidian(a, b):
    return torch.linalg.norm(a - b)

def squared_euclidian_pairwise_distances(X: torch.Tensor):
    """
    Computes pairwise euclidian distance between every pair of points 
    in the sample matrix X

    Returns the square of the euclician distance
    """
    return torch.sum((X[None, :] - X[:, None]) ** 2, axis=2)

def euclidian_pairwise_distances(X: torch.Tensor):
    """
    Computes pairwise euclidian distance between every pair of points 
    in the sample matrix X
    """
    return torch.sqrt(squared_euclidian_pairwise_distances(X))
