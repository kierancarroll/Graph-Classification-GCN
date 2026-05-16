# Training loop

from typing import Tuple
import torch
import torch.nn as nn

def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    train_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    valid_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    log_every: int = 500):
    """Train a GraphClassifier and track loss and validation accuracy.

    Args:
        model:      The model to train.
        optimizer:  Optimiser instance (e.g. Adam).
        criterion:  Loss function (e.g. BCEWithLogitsLoss).
        epochs:     Number of full training passes.
        train_data: Tuple of (a_norm, features, batch_idx, labels) for training.
        valid_data: Tuple of (a_norm, features, batch_idx, labels) for validation.
        log_every:  Print a progress line every this many epochs.

    Returns:
        train_losses: Loss value recorded after each epoch.
        val_accuracies: Validation accuracy recorded after each epoch.
    """
    from test import evaluate  # local import to avoid circular dependency

    a_norm_tr, features_tr, batch_idx_tr, labels_tr = train_data
    train_losses = []
    val_accuracies = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(a_norm_tr, features_tr, batch_idx_tr)
        loss = criterion(logits, labels_tr)
        loss.backward()
        optimizer.step()

        val_acc = evaluate(model, valid_data)

        train_losses.append(loss.item())
        val_accuracies.append(val_acc)

        if epoch % log_every == 0:
            print(
                f"Epoch {epoch:>5d}/{epochs} | "
                f"Train Loss: {loss.item():.4f} | "
                f"Val Accuracy: {val_acc:.4f}"
            )

    return train_losses, val_accuracies