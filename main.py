"""Main file for GCN graph classification on the PROTEINS dataset.

Runs four experiments (baseline, dropout, L2, dropout + L2) for a configurable number of trials 
and prints a summary of mean ± std test accuracy and plots training loss and validationa accuracy across epochs.
"""

import numpy as np
import yaml
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import torch
import torch.nn as nn
import torch.optim as optim

from data import load_proteins, split_dataset
from utils.batch import prepare_batch
from models.graph_classifier import GraphClassifier
from train import train
from test import evaluate


# Configurations
with open(f"config.yaml", "r") as f:
    config = yaml.safe_load(f)

    EPOCHS = config["training"]["epochs"]
    HIDDEN_DIM = config["model"]["hidden_dim"]
    LR = float(config["model"]["lr"])
    NUM_TRIALS = config["training"]["num_trials"]
    LOG_EVERY = config["training"]["log_every"]
    MOVING_AVG_WINDOW = config["training"]["moving_avg_window"]

EXPERIMENTS = {
    "baseline":   {"dropout_p": 0.0, "weight_decay": 0.0},
    "dropout":    {"dropout_p": 0.5, "weight_decay": 0.0},
    "l2":         {"dropout_p": 0.0, "weight_decay": 1e-4},
    "dropout+l2": {"dropout_p": 0.5, "weight_decay": 1e-4},
}

COLORS = {
    "baseline":   "#1f77b4",
    "dropout":    "#ff7f0e",
    "l2":         "#2ca02c",
    "dropout+l2": "#d62728",
}

# Plotting helpers functions
def moving_average(values: list, window: int = 50) -> np.ndarray:
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_comparison(all_losses: dict, all_val_accs: dict, epochs: int) -> None:
    """Plot averaged training loss and validation accuracy for all experiments."""
    ma_x = np.arange(1, epochs - MOVING_AVG_WINDOW + 2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for name in EXPERIMENTS:
        avg_loss = np.mean(all_losses[name], axis=0)
        avg_acc = np.mean(all_val_accs[name], axis=0)

        ax1.plot(ma_x, moving_average(avg_loss, MOVING_AVG_WINDOW),
                 label=name, color=COLORS[name], linewidth=2)
        ax2.plot(ma_x, moving_average(avg_acc, MOVING_AVG_WINDOW),
                 label=name, color=COLORS[name], linewidth=2)

    ax1.set_title(f"Training Loss (averaged over {NUM_TRIALS} trials, with {MOVING_AVG_WINDOW} epoch moving avg)", fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCE Loss")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()

    ax2.set_title(f"Validation Accuracy (averaged over {NUM_TRIALS} trials, with {MOVING_AVG_WINDOW} epoch moving avg)", fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("figures/results_curves.png", dpi=150)
    plt.show()


def plot_test_accuracy(test_results: dict) -> None:
    """Bar chart of mean ± std test accuracy per experiment."""
    names = list(test_results.keys())
    means = [np.mean(test_results[n]) for n in names]
    stds = [np.std(test_results[n]) for n in names]
    colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(names, means, yerr=stds, color=colors, alpha=0.75,
                  edgecolor="black", capsize=8)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean:.4f}\n±{std:.4f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_title(f"Test Accuracy — Mean ± Std over {NUM_TRIALS} trials", fontweight="bold")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(max(0, min(means) - max(stds) - 0.05), min(1, max(means) + max(stds) + 0.07))
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("figures/results_test_accuracy.png", dpi=150)
    plt.show()

def run_tsne_experiment(train_data, valid_data, test_data, num_features, device):
    print(f"\n{'='*60}")
    print("  t-SNE EXPERIMENT (baseline model)")
    print(f"{'='*60}")

    # same hyperparams as baseline
    model = GraphClassifier(input_features=num_features, hidden_dim=HIDDEN_DIM, num_classes=1, dropout_p=0.0).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=0.0)
    criterion = nn.BCEWithLogitsLoss()

    # train once
    losses, val_accs = train(
        model,
        optimizer,
        criterion,
        EPOCHS,
        train_data,
        valid_data,
        log_every=LOG_EVERY,
    )

    # evaluate + extract embeddings
    model.eval()
    a_hat, x, batch_idx, labels = test_data

    with torch.no_grad():
        logits, embeddings = model(a_hat, x, batch_idx, return_embeddings=True)

        preds = (torch.sigmoid(logits) > 0.5).long()
        acc = (preds == labels).float().mean().item()

    print(f"Test accuracy: {acc:.4f}")

    # convert to numpy
    embeddings = embeddings.cpu().numpy()
    labels = labels.cpu().numpy()

    # t-SNE
    # tsne = TSNE(n_components=2, random_state=42)
    # emb_2d = tsne.fit_transform(embeddings)
    emb_2d = umap.UMAP().fit_transform(embeddings)

    # plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        emb_2d[:, 0],
        emb_2d[:, 1],
        c=labels,
        cmap="coolwarm",
        s=35,
        alpha=0.8,
    )
    plt.title("t-SNE of Graph Embeddings (Baseline GCN)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.colorbar(label="Class")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("figures/tsne_graph_embeddings.png", dpi=150)
    plt.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}\n")

    print("Loading and preprocessing dataset...")
    dataset = load_proteins()
    train_ds, valid_ds, test_ds = split_dataset(dataset)

    print("Preparing batches (building block-diagonal adjacency matrices)...")
    train_data = tuple(t.to(device) for t in prepare_batch(train_ds))
    valid_data = tuple(t.to(device) for t in prepare_batch(valid_ds))
    test_data  = tuple(t.to(device) for t in prepare_batch(test_ds))

    num_features = train_data[1].shape[1]
    criterion = nn.BCEWithLogitsLoss()

    # experiments 
    all_losses = {name: [] for name in EXPERIMENTS}
    all_val_accs = {name: [] for name in EXPERIMENTS}
    test_results = {name: [] for name in EXPERIMENTS}

    train_loss_results = {name: [] for name in EXPERIMENTS}
    val_acc_results = {name: [] for name in EXPERIMENTS}

    # for trial in range(1, NUM_TRIALS + 1):
    #     print(f"\n{'='*60}")
    #     print(f"  TRIAL {trial}/{NUM_TRIALS}")
    #     print(f"{'='*60}")

    #     for name, cfg in EXPERIMENTS.items():
    #         print(f"\n--- {name.upper()} ---")
    #         model = GraphClassifier(
    #             input_features=num_features,
    #             hidden_dim=HIDDEN_DIM,
    #             num_classes=1,
    #             dropout_p=cfg["dropout_p"],
    #         ).to(device)

    #         optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=cfg["weight_decay"])

    #         losses, val_accs = train(
    #             model, optimizer, criterion,
    #             EPOCHS, train_data, valid_data,
    #             log_every=LOG_EVERY,
    #         )

    #         test_acc = evaluate(model, test_data)
    #         print(f"Test accuracy: {test_acc:.4f}")

    #         all_losses[name].append(losses)
    #         all_val_accs[name].append(val_accs)
    #         test_results[name].append(test_acc)

    #         train_loss_results[name].append(np.mean(losses[-10:]))   
    #         val_acc_results[name].append(np.max(val_accs))    

    # # summary
    # print(f"\n{'='*60}")
    # print(f"  SUMMARY  (mean ± std over {NUM_TRIALS} trials)")
    # print(f"{'='*60}")

    # for name in EXPERIMENTS:
    #     test_mean = np.mean(test_results[name])
    #     test_std  = np.std(test_results[name])

    #     train_mean = np.mean(train_loss_results[name])
    #     train_std  = np.std(train_loss_results[name])

    #     val_mean = np.mean(val_acc_results[name])
    #     val_std  = np.std(val_acc_results[name])

    #     print(
    #         f"  {name:<14s}  "
    #         f"Test: {test_mean:.4f} ± {test_std:.4f} | "
    #         f"Train Loss: {train_mean:.4f} ± {train_std:.4f} | "
    #         f"Val Acc: {val_mean:.4f} ± {val_std:.4f}"
    #     )

    # # plots
    # plot_comparison(all_losses, all_val_accs, EPOCHS)
    # plot_test_accuracy(test_results)

    # t-SNE visualization
    run_tsne_experiment(train_data, valid_data, test_data, num_features, device)

if __name__ == "__main__":
    main()