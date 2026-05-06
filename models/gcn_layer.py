import torch
import torch.nn as nn


class GCNLayer(nn.Module):
    ''' Single Graph Convolutional Network layer.

    Implements the propagation rule from Kipf & Welling (2017): Z = A_hat @ X @ W
     where A_hat is the pre-computed normalised adjacency matrix (with added
     self-loops) and W is a learnable weight matrix initialised with Kaiming
     uniform initialisation.
    '''

    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_features, output_features))
        nn.init.kaiming_uniform_(self.weight)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        return torch.spmm(a_hat, x @ self.weight)