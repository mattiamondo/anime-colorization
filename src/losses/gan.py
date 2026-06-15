"""Adversarial loss wrapper for PatchGAN outputs."""

import torch
import torch.nn as nn


class GANLoss(nn.Module):
    """Adversarial loss on patch logit maps.

    Args:
        mode: "vanilla" (BCEWithLogits, Pix2Pix) or "lsgan" (MSE, CycleGAN).
    """

    def __init__(self, mode: str = "vanilla"):
        super().__init__()
        if mode == "vanilla":
            self.criterion = nn.BCEWithLogitsLoss()
        elif mode == "lsgan":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(f"mode must be 'vanilla' or 'lsgan', got {mode!r}")

    def forward(self, prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        target = torch.full_like(prediction, float(target_is_real))
        return self.criterion(prediction, target)
