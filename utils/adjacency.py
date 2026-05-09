# Normalised adjacency matrix computation for GCN preprocessing

import numpy as np
import scipy.sparse
import torch
from scipy.sparse import diags


def compute_a_norm(data_obj) -> scipy.sparse.spmatrix:
    """Compute the normalised adjacency matrix A_hat for a single graph.

    Args:
        data_obj: A PyG `Data` object for a single graph.

    Returns:
        A (num_nodes × num_nodes) scipy sparse matrix in float32.
    """
    edge_index = data_obj.edge_index
    rows = edge_index[0].numpy()
    cols = edge_index[1].numpy()
    values = np.ones(len(rows), dtype=np.float32)

    adj = scipy.sparse.coo_matrix(
        (values, (rows, cols)),
        shape=(data_obj.num_nodes, data_obj.num_nodes),
        dtype=np.float32,
    )

    adj_tilde = adj + diags(np.ones(data_obj.num_nodes, dtype=np.float32))
    degrees = adj_tilde.sum(axis=1).A1
    d_inv_sqrt = diags(degrees, dtype=np.float32).power(-0.5)

    return d_inv_sqrt @ adj_tilde @ d_inv_sqrt


def sparse_to_torch(matrix: scipy.sparse.spmatrix) -> torch.Tensor:
    """Convert a scipy sparse matrix to a sparse PyTorch float tensor.

    Args:
        matrixt: Any scipy sparse matrix (will be converted to COO internally).

    Returns:
        A torch.sparse_coo_tensor of dtype float32.
    """
    matrix = matrix.tocoo()
    indices = np.vstack([matrix.row, matrix.col])
    return torch.sparse_coo_tensor(torch.from_numpy(indices).long(), torch.from_numpy(matrix.data), matrix.shape, dtype=torch.float)