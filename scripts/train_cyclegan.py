#!/usr/bin/env python
"""Headless training runner for variant 4 (CycleGAN, unpaired).

Designed for unattended overnight runs on the shared GPU server:

- Resumes from ``checkpoints/cyclegan/last.pt`` if present, so it continues from
  the last completed epoch instead of restarting.
- Meant to be wrapped in a shell auto-retry loop (see README), so a transient
  CUDA failure caused by the other user's jobs on the shared GPU just restarts
  the script, which resumes from the last checkpoint.
- ``print`` / errors go straight to stdout/stderr, so the real cause of any
  crash is visible live in the log (unlike ``nbconvert``, which only reports a
  generic "Kernel died").

The notebook ``notebooks/04_cyclegan.ipynb`` loads the checkpoints this script
produces to present the loss curves, the qualitative grid and the metrics — it
never needs to run the long training itself.

Usage (inside the conda env, in a tmux session):
    python scripts/train_cyclegan.py
"""

import os

# Must be set before torch initializes CUDA. Honors an externally provided
# CUDA_VISIBLE_DEVICES so you can pick the free GPU at launch on this shared
# server (e.g. `CUDA_VISIBLE_DEVICES=0 python scripts/train_cyclegan.py`).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
# Reduce fragmentation-related OOMs when sharing the GPU with another user.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader

from src.data import AnimeColorizationDataset
from src.training import CycleGANTrainer
from src.utils import seed_everything

# Keep this CONFIG in sync with notebooks/04_cyclegan.ipynb.
CONFIG = dict(
    image_size=256,
    batch_size=8,
    num_workers=4,
    epochs=50,
    decay_start=25,
    lr=2e-4,
    lambda_cycle=10.0,
    lambda_identity=5.0,
    use_amp=True,
)

DATA_ROOT = ROOT / "data" / "anime_colorization"
CKPT_DIR = ROOT / "checkpoints" / "cyclegan"


def main() -> None:
    seed_everything(42)

    # Training uses paired=False (unpaired setting); val uses paired=True so the
    # trainer can monitor val_l1 against ground truth for checkpoint selection.
    train_ds = AnimeColorizationDataset(
        str(DATA_ROOT), split="train",
        image_size=CONFIG["image_size"], paired=False)
    val_ds = AnimeColorizationDataset(
        str(DATA_ROOT), split="val",
        image_size=CONFIG["image_size"], paired=True)

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True)

    print(f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} | "
          f"train={len(train_ds):,} val={len(val_ds):,} | "
          f"batch_size={CONFIG['batch_size']}", flush=True)

    trainer = CycleGANTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=str(CKPT_DIR),
        monitor="val_l1",
        device="cuda",
        lambda_cycle=CONFIG["lambda_cycle"],
        lambda_identity=CONFIG["lambda_identity"],
        lr=CONFIG["lr"],
        decay_start=CONFIG["decay_start"],
        use_amp=CONFIG["use_amp"],
    )

    # Resume if a previous (possibly crashed) run left a checkpoint.
    if (CKPT_DIR / "last.pt").exists():
        trainer.load_checkpoint("last.pt")
        print(f"Resumed from epoch {trainer.epoch} "
              f"(best val_l1={trainer.best_metric:.4f})", flush=True)

    if trainer.epoch >= CONFIG["epochs"]:
        print(f"Already trained {trainer.epoch}/{CONFIG['epochs']} epochs — "
              f"nothing to do.", flush=True)
        return

    trainer.fit(CONFIG["epochs"])
    print("Training complete.", flush=True)


if __name__ == "__main__":
    main()
