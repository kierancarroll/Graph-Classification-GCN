# PROTEINS dataset loading and train/val/test split creation.

import math
import torch
from torch_geometric.datasets import TUDataset


def load_proteins(root: str = "data/TU") -> TUDataset:
    """Download (if needed) and return the full PROTEINS dataset.

    Args:
        root: Directory where the dataset will be cached.

    Returns:
        The full TUDataset with 1,113 protein graphs (proteins).
    """
    return TUDataset(root=root, name="PROTEINS", use_node_attr=True)


def split_dataset(dataset: TUDataset, seed: int = 42, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Randomly split a dataset into train, validation, and test subsets (default 80/10/10 split).

    Args:
        dataset:     The full dataset to split.
        seed:        Random seed for reproducibility.
        train_ratio: Fraction of data used for training.
        val_ratio:   Fraction of data used for validation.
                     The remainder goes to the test set.

    Returns:
        Tuple of (train_dataset, valid_dataset, test_dataset).
    """
    torch.random.manual_seed(seed)
    indices = torch.randperm(len(dataset))

    n_train = math.floor(train_ratio * len(dataset))
    n_val = math.floor(val_ratio * len(dataset))

    train_dataset = dataset[indices[:n_train]]
    valid_dataset = dataset[indices[n_train : n_train + n_val]]
    test_dataset = dataset[indices[n_train + n_val :]]

    return train_dataset, valid_dataset, test_dataset