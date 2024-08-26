import torch


def euclidian(a, b):
    return torch.linalg.norm(a - b)
