"""Batch preparation for processing multiple graphs in a single GCN pass."""

from typing import Tuple

import scipy.sparse
import torch
from .adjacency import compute_a_norm, sparse_to_torch


def prepare_batch(dataset) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build batched representation for a collection of protein graphs.

    Args:
        dataset: An iterable of PyG `Data` objects (e.g. a TUDataset split).

    Returns:
        a_norm    (torch.sparse_coo_tensor): Block-diagonal A_hat.
        features  (torch.Tensor):           Concatenated node features.
        batch_idx (torch.Tensor):           Node-to-graph index mapping (long).
        labels    (torch.Tensor):           Graph labels as float32.
    """
    adj_matrices = []
    feature_matrices = []
    batch_indices = []
    labels = []

    for graph_id, data_obj in enumerate(dataset):
        a_norm = compute_a_norm(data_obj)
        adj_matrices.append(a_norm)
        feature_matrices.append(data_obj.x)
        batch_indices.append(torch.full((data_obj.num_nodes,), graph_id, dtype=torch.long))
        labels.append(data_obj.y.float())

    block_diag = scipy.sparse.block_diag(adj_matrices)
    a_norm = sparse_to_torch(block_diag)
    features = torch.cat(feature_matrices, dim=0)
    batch_idx = torch.cat(batch_indices, dim=0)
    labels = torch.cat(labels, dim=0)

    return a_norm, features, batch_idx, labels