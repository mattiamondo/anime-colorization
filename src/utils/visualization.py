"""Plotting helpers: loss curves and qualitative comparison grids."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def _denorm(t: torch.Tensor) -> np.ndarray:
    """[-1, 1] CHW tensor -> [0, 1] HWC array for imshow."""
    return (t.detach().cpu().permute(1, 2, 0).numpy() + 1) / 2


def _save(fig, save_path: str | None) -> None:
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)


def plot_loss_curves(history: dict[str, list[float]], title: str = "",
                     save_path: str | None = None):
    """Plot train/val loss curves in separate subplots."""
    train_keys = [k for k in history if k.startswith("train_")]
    val_keys   = [k for k in history if k.startswith("val_")]
    epochs = range(1, len(next(iter(history.values()))) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key in train_keys:
        axes[0].plot(epochs, history[key], label=key, linewidth=2)
    for key in val_keys:
        axes[1].plot(epochs, history[key], label=key, linewidth=2)
    for ax, subtitle in zip(axes, ("Train losses", "Val losses")):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (log scale)")
        ax.set_yscale("log")
        ax.set_title(subtitle)
        ax.legend()
        ax.grid(alpha=0.3, which="both")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, save_path)
    plt.show()


def qualitative_grid(sketch: torch.Tensor, prediction: torch.Tensor,
                     target: torch.Tensor, n_rows: int = 4,
                     save_path: str | None = None):
    """Grid with columns: sketch | prediction | ground truth"""
    n_rows = min(n_rows, sketch.shape[0])
    fig, axes = plt.subplots(n_rows, 3, figsize=(9, 3 * n_rows))
    axes = np.atleast_2d(axes)
    for row in range(n_rows):
        for col, (name, batch) in enumerate(
                [("sketch", sketch), ("prediction", prediction),
                 ("ground truth", target)]):
            ax = axes[row, col]
            ax.imshow(_denorm(batch[row]))
            ax.axis("off")
            if row == 0:
                ax.set_title(name)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def multi_model_grid(sketch: torch.Tensor, predictions: dict[str, torch.Tensor],
                     target: torch.Tensor, save_path: str | None = None):
    """Grid comparing all variants on the same sketches.

    Columns: sketch | variant 1 | ... | variant 5 | ground truth.
    """
    columns = [("sketch", sketch), *predictions.items(), ("ground truth", target)]
    n_rows, n_cols = sketch.shape[0], len(columns)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.2 * n_rows))
    axes = np.atleast_2d(axes)
    for row in range(n_rows):
        for col, (name, batch) in enumerate(columns):
            ax = axes[row, col]
            ax.imshow(_denorm(batch[row]))
            ax.axis("off")
            if row == 0:
                ax.set_title(name)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()
