import torch
import torch.nn as nn
from .gcn_layer import GCNLayer


class GCN(nn.Module):
    """Two-layer Graph Convolutional Network.

    Architecture:
        GCNLayer → ReLU → Dropout → GCNLayer

    Arguments:
        input_features:  Dimensionality of the raw node features.
        hidden_features: Width of the hidden layer.
        output_features: Dimensionality of the final node embeddings.
        dropout:         Dropout probability applied between the two layers
                         (default: 0.0, i.e. no dropout).
    """

    def __init__(self, input_features: int, hidden_features: int, output_features: int, dropout: float = 0.0):
        super().__init__()
        self.layer1 = GCNLayer(input_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, output_features)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        if a_hat.device != x.device:
            a_hat = a_hat.to(x.device)

        x = self.relu(self.layer1(x, a_hat))
        x = self.dropout(x)
        return self.layer2(x, a_hat)