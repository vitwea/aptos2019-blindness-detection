import numpy as np
from torch.utils.data import WeightedRandomSampler

def get_balanced_sampler(labels):
    """
    Creates a WeightedRandomSampler to balance class frequencies.
    labels: array-like of class labels
    """
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts

    sample_weights = class_weights[labels]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler