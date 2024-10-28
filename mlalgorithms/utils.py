from typing import List
import torch

def gini(splits: List, labels):
    """
    Computes the Gini index of a split given the labels

    Args :
    - splits    : List of arrays detailing the splits
    - labels    : Labels given by the split

    Returns the gini index of this split
    """
    split_counts = [len(split) for split in splits]
    nb_samples = sum(split_counts)

    gini = 0
    for i, split in enumerate(splits):
        split_count = split_counts[i]
        if split_count == 0:
            continue
        
        split_score = 0
        for label in labels:
            split_score += (torch.sum(torch.Tensor(split) == torch.Tensor(label)) / split_count) ** 2
        gini += (1 - split_score) / (split_count / nb_samples)
        
    return gini
