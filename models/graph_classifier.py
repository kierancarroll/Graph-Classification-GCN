import torch
import torch.nn as nn
from torch_scatter import scatter
from .gcn import GCN


class GraphClassifier(nn.Module):
    """GCN-based binary graph classifier.

    Architecture:
        GCN → ReLU → scatter-max pooling → Linear

    A batch of graphs is processed in a single forward pass by representing
    the batch as a block-diagonal adjacency matrix and a concatenated feature
    matrix. After the GCN produces per-node embeddings, scatter-max pooling
    reduces each graph's nodes to a single fixed-size vector, which is then
    fed into a linear classifier.

    Arguments:
        input_features: Dimensionality of the raw node features.
        hidden_dim:     Width of the GCN hidden layer and the pooled
                        graph-level embedding.
        num_classes:    Number of output logits (default: 1 for binary
                        classification with BCEWithLogitsLoss).
        dropout_p:      Dropout probability inside the GCN (default: 0.0).
    """

    def __init__(self, input_features: int, hidden_dim: int, num_classes: int = 1, dropout_p: float = 0.0):
        super().__init__()
        self.gcn = GCN(input_features, hidden_dim, hidden_dim, dropout=dropout_p)
        self.linear = nn.Linear(hidden_dim, num_classes)

    def forward(self, a_hat: torch.Tensor, x: torch.Tensor, batch_idx: torch.Tensor, return_embeddings: bool = False) -> torch.Tensor:
        x = self.gcn(x, a_hat)
        x = torch.relu(x)
        graph_embeddings = scatter(x, batch_idx, dim=0, reduce="max")
        logits = self.linear(graph_embeddings).squeeze(-1)
        if return_embeddings:
            return logits, graph_embeddings
        else:
            return logits 