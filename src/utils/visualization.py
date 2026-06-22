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
                     save_path: str | None = None) -> None:
    """Plot train/val loss curves with variant-aware panel layout.

    Automatically detects Pix2Pix vs CycleGAN from history keys and splits
    curves into dedicated panels (generator components / discriminator / val).
    ``train_g`` (weighted total generator loss) is excluded from Pix2Pix plots
    because it is redundant with the individual unweighted component losses.
    CycleGAN keeps ``train_G`` since the adversarial component is not logged
    separately and the total loss is the only generator-level trend available.
    """
    _LABELS: dict[str, str] = {
        "train_l1":         "L₁ (generator)",
        "train_gan":        "GAN (generator)",
        "train_perceptual": "Perceptual (generator)",
        "train_d":          "Discriminator",
        "train_G":          "Total G loss",
        "train_D_color":    "Disc. (color)",
        "train_D_sketch":   "Disc. (sketch)",
        "train_cycle":      "Cycle-consistency",
        "train_idt":        "Identity",
        "val_l1":           "Val L₁",
    }

    val_keys = [k for k in history if k.startswith("val_")]

    if "train_cycle" in history:
        # CycleGAN: 3 panels — generator | discriminators | val
        gen_keys  = [k for k in ("train_G", "train_cycle", "train_idt") if k in history]
        disc_keys = [k for k in ("train_D_color", "train_D_sketch") if k in history]
        panels = [
            (gen_keys,  "Generator losses"),
            (disc_keys, "Discriminator losses"),
            (val_keys,  "Val losses"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        # Pix2Pix family: exclude train_g
        gen_keys  = [k for k in ("train_l1", "train_gan", "train_perceptual")
                     if k in history]
        disc_keys = [k for k in ("train_d",) if k in history]
        if disc_keys:
            # Variants with discriminator: 3 panels
            panels = [
                (gen_keys,  "Generator losses"),
                (disc_keys, "Discriminator loss"),
                (val_keys,  "Val losses"),
            ]
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            # V1 — no adversarial component: 2 panels
            panels = [
                (gen_keys, "Train losses"),
                (val_keys, "Val losses"),
            ]
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(next(iter(history.values()))) + 1)

    for ax, (keys, subtitle) in zip(axes, panels):
        for key in keys:
            ax.plot(epochs, history[key], label=_LABELS.get(key, key), linewidth=2)
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
                     title: str | None = None,
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
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def qualitative_grid_compare(sketch: torch.Tensor,
                             pred_last: torch.Tensor, pred_best: torch.Tensor,
                             target: torch.Tensor,
                             epoch_last: int, epoch_best: int,
                             n_rows: int = 4,
                             save_path: str | None = None):
    """4-column grid: sketch | last.pt (ep N) | best.pt (ep M) | ground truth."""
    n_rows = min(n_rows, sketch.shape[0])
    columns = [
        (f"sketch", sketch),
        (f"last.pt  (ep. {epoch_last})", pred_last),
        (f"best.pt  (ep. {epoch_best})", pred_best),
        ("ground truth", target),
    ]
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))
    axes = np.atleast_2d(axes)
    for row in range(n_rows):
        for col, (name, batch) in enumerate(columns):
            ax = axes[row, col]
            ax.imshow(_denorm(batch[row]))
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)
    plt.show()


def cycle_grid(sketch: torch.Tensor, fake_color: torch.Tensor,
               rec_sketch: torch.Tensor, target: torch.Tensor,
               n_rows: int = 4, save_path: str | None = None):
    """Forward-cycle visualization for CycleGAN: sketch → fake color → rec. sketch.

    Columns: sketch (input) | G_s2c(sketch) | G_c2s(G_s2c(sketch)) | ground truth.
    Shows that cycle-consistency is satisfied even when colorization is wrong.
    """
    n_rows = min(n_rows, sketch.shape[0])
    columns = [
        ("sketch  (input)", sketch),
        ("G_s2c →  fake color", fake_color),
        ("G_c2s →  rec. sketch", rec_sketch),
        ("ground truth", target),
    ]
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 3 * n_rows))
    axes = np.atleast_2d(axes)
    for row in range(n_rows):
        for col, (name, batch) in enumerate(columns):
            ax = axes[row, col]
            ax.imshow(_denorm(batch[row]))
            ax.axis("off")
            if row == 0:
                ax.set_title(name, fontsize=9)
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
