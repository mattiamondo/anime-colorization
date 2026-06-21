#!/usr/bin/env python
"""Headless training runner for variant 4 (CycleGAN, unpaired).

Trained WITHOUT pairing information: sketch and color images are sampled
independently (paired=False in the dataset). Pairs exist but the model never
sees them during training — this intentionally simulates the unpaired setting
to quantify the value of paired supervision (Pix2Pix variants 2/3) vs unpaired.

Validation and checkpoint selection use paired=True so best.pt tracks the
sketch→color generator quality against ground-truth colors.

Usage (inside the conda env, in a tmux session):
    python scripts/train_04_cyclegan.py
    python scripts/train_04_cyclegan.py 2>&1 | tee results/logs/04_cyclegan.log
    CUDA_VISIBLE_DEVICES=N python scripts/train_04_cyclegan.py
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
from src.training import CycleGANTrainer
from src.utils import seed_everything

# Keep in sync with notebooks/04_cyclegan.ipynb.
# batch_size=8: CycleGAN has 4 networks (2 G + 2 D), much heavier than Pix2Pix.
# decay_start=50: constant LR for first 50 epochs, linear decay to 0 for the rest.
CONFIG = dict(
    image_size=256,
    batch_size=8,
    num_workers=8,
    epochs=100,
    decay_start=50,
    lr=2e-4,
    lambda_cycle=10.0,
    lambda_identity=5.0,
    use_amp=True,
    early_stopping=False,
)

DATA_ROOT = ROOT / "data" / "anime_colorization"
CKPT_DIR  = ROOT / "checkpoints" / "04_cyclegan"


def main() -> None:
    seed_everything(42)

    train_ds = AnimeColorizationDataset(
        str(DATA_ROOT), split="train", image_size=CONFIG["image_size"], paired=False)
    val_ds = AnimeColorizationDataset(
        str(DATA_ROOT), split="val", image_size=CONFIG["image_size"], paired=True)

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
        f"train={len(train_ds):,} (unpaired) val={len(val_ds):,} (paired) | "
        f"batch_size={CONFIG['batch_size']}",
        flush=True,
    )

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
        early_stopping=CONFIG["early_stopping"],
        wandb_project="anime-colorization",
        wandb_run_name="04-cyclegan-unpaired",
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
