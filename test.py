# Evaluation (validation & test) for graph classification

from typing import Tuple
import torch
import torch.nn as nn

@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
    """Compute binary classification accuracy on a dataset split.

    Argumentss:
        model: A trained GraphClassifier.
        data:  Tuple of (a_norm, features, batch_idx, labels).
        
    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    a_norm, features, batch_idx, labels = data
    logits = model(a_norm, features, batch_idx)
    #Logits are passed through sigmoid and thresholded at 0.5 to obtain predicted class labels.
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    accuracy = (preds == labels).float().mean().item()
    return accuracy