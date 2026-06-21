#!/usr/bin/env python
"""Headless training runner for variant 1 (U-Net + L1 baseline).

Pure L1 regression: U-Net encoder-decoder with skip connections, no adversarial
or perceptual loss. Serves as the lower-bound baseline in the ablation study.

Usage (inside the conda env, in a tmux session):
    python scripts/train_01_unet_l1.py
    python scripts/train_01_unet_l1.py 2>&1 | tee results/logs/01_unet_l1.log
    CUDA_VISIBLE_DEVICES=N python scripts/train_01_unet_l1.py
"""

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from torch.utils.data import DataLoader

from src.data import AnimeColorizationDataset
from src.training import Pix2PixTrainer
from src.utils import seed_everything

# Keep in sync with notebooks/01_unet_l1_baseline.ipynb
CONFIG = dict(
    image_size=256,
    batch_size=16,
    num_workers=12,
    epochs=100,
    lr=2e-4,
    lambda_l1=100.0,
    lambda_gan=0.0,        # no discriminator -> variant 1
    lambda_perceptual=0.0,
    patience=20,
    early_stopping=True,   # val_l1 is the correct monitor for pure L1
)

DATA_ROOT = ROOT / "data" / "anime_colorization"
CKPT_DIR  = ROOT / "checkpoints" / "01_unet_l1"


def main() -> None:
    seed_everything(42)

    train_ds = AnimeColorizationDataset(DATA_ROOT, split="train", image_size=CONFIG["image_size"])
    val_ds   = AnimeColorizationDataset(DATA_ROOT, split="val",   image_size=CONFIG["image_size"])

    train_loader = DataLoader(
        train_ds, batch_size=CONFIG["batch_size"], shuffle=True,
        num_workers=CONFIG["num_workers"], pin_memory=True,
        drop_last=True, persistent_workers=True)
    val_loader = DataLoader(
        val_ds, batch_size=CONFIG["batch_size"], shuffle=False,
        num_workers=CONFIG["num_workers"], pin_memory=True,
        persistent_workers=True)

    print(
        f"CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} | "
        f"train={len(train_ds):,} val={len(val_ds):,} | "
        f"batch_size={CONFIG['batch_size']}",
        flush=True,
    )

    trainer = Pix2PixTrainer(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=str(CKPT_DIR),
        monitor="val_l1",
        device="cuda",
        lambda_l1=CONFIG["lambda_l1"],
        lambda_gan=CONFIG["lambda_gan"],
        lambda_perceptual=CONFIG["lambda_perceptual"],
        lr=CONFIG["lr"],
        patience=CONFIG["patience"],
        early_stopping=CONFIG["early_stopping"],
        wandb_project="anime-colorization",
        wandb_run_name="01-unet-l1-baseline",
        wandb_config=CONFIG,
        wandb_log_images_every=5,
    )

    if (CKPT_DIR / "last.pt").exists():
        trainer.load_checkpoint("last.pt")
        print(
            f"Resumed from epoch {trainer.epoch} "
            f"(best val_l1={trainer.best_metric:.4f})",
            flush=True,
        )

    if trainer.epoch >= CONFIG["epochs"]:
        print(
            f"Already trained {trainer.epoch}/{CONFIG['epochs']} epochs — nothing to do.",
            flush=True,
        )
        trainer.finish()
        return

    trainer.fit(CONFIG["epochs"])
    trainer.finish()
    print(f"Training complete. Checkpoints in: {CKPT_DIR}", flush=True)


if __name__ == "__main__":
    main()
