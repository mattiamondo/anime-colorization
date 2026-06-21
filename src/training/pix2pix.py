"""Trainer for the paired variants 1-3.

A single trainer covers three ablation steps via loss weights:
- variant 1 (U-Net baseline):    lambda_gan=0, lambda_perceptual=0
- variant 2 (Pix2Pix):           lambda_gan>0, lambda_perceptual=0
- variant 3 (Pix2Pix+perceptual) lambda_gan>0, lambda_perceptual>0
"""

import torch
import torch.nn as nn
from torchvision.utils import make_grid

from ..losses import GANLoss, VGGPerceptualLoss
from ..models import GlobalDiscriminator, PatchGANDiscriminator, UNetGenerator
from .base import BaseTrainer


class Pix2PixTrainer(BaseTrainer):
    """U-Net generator + optional discriminator + optional perceptual loss.

    Args:
        lambda_l1: weight of the pixel-wise L1 loss (100 in the paper).
        lambda_gan: weight of the adversarial loss (0 disables the
            discriminator entirely -> variant 1).
        lambda_perceptual: weight of the VGG perceptual loss (0 -> off).
        discriminator_type: "patch" (70×70 PatchGAN, default) or "global"
            (single-scalar global discriminator, variant 6).
        lr / betas: Adam hyperparameters (paper defaults).
    """

    def __init__(self, *, lambda_l1: float = 100.0, lambda_gan: float = 1.0,
                 lambda_perceptual: float = 0.0,
                 discriminator_type: str = "patch",
                 lr: float = 2e-4,
                 betas: tuple[float, float] = (0.5, 0.999), **kwargs):
        super().__init__(**kwargs)
        self.lambda_l1 = lambda_l1
        self.lambda_gan = lambda_gan
        self.lambda_perceptual = lambda_perceptual

        self.generator = self.register_network("generator", UNetGenerator())
        self.optimizers["generator"] = torch.optim.Adam(
            self.generator.parameters(), lr=lr, betas=betas)
        self.l1_loss = nn.L1Loss()

        if lambda_gan > 0:
            if discriminator_type == "global":
                disc = GlobalDiscriminator(in_channels=6)
            elif discriminator_type == "patch":
                disc = PatchGANDiscriminator(in_channels=6)
            else:
                raise ValueError(
                    f"discriminator_type must be 'patch' or 'global', "
                    f"got {discriminator_type!r}")
            self.discriminator = self.register_network("discriminator", disc)
            self.optimizers["discriminator"] = torch.optim.Adam(
                self.discriminator.parameters(), lr=lr, betas=betas)
            self.gan_loss = GANLoss("vanilla").to(self.device)

        if lambda_perceptual > 0:
            # No need for register_network because VGG is frozen
            self.perceptual_loss = VGGPerceptualLoss().to(self.device)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        sketch, color = batch["sketch"], batch["color"]
        fake = self.generator(sketch)
        logs: dict[str, float] = {}

        if self.lambda_gan > 0:
            # Discriminator training
            opt_d = self.optimizers["discriminator"]
            opt_d.zero_grad(set_to_none=True)
            pred_real = self.discriminator(torch.cat([sketch, color], dim=1))
            pred_fake = self.discriminator(torch.cat([sketch, fake.detach()], dim=1))
            loss_d = 0.5 * (self.gan_loss(pred_real, True)
                            + self.gan_loss(pred_fake, False))
            loss_d.backward()
            opt_d.step()
            logs["train_d"] = loss_d.item()

        # Generator training (without discriminator)
        opt_g = self.optimizers["generator"]
        opt_g.zero_grad(set_to_none=True)
        loss_l1 = self.l1_loss(fake, color)
        loss_g = self.lambda_l1 * loss_l1
        logs["train_l1"] = loss_l1.item()

        # Generator training (with discriminator)
        if self.lambda_gan > 0:
            pred_fake = self.discriminator(torch.cat([sketch, fake], dim=1))
            loss_gan = self.gan_loss(pred_fake, True)
            loss_g = loss_g + self.lambda_gan * loss_gan
            logs["train_gan"] = loss_gan.item()

        # Generator training (with perceptual)
        if self.lambda_perceptual > 0:
            loss_perc = self.perceptual_loss(fake, color)
            loss_g = loss_g + self.lambda_perceptual * loss_perc
            logs["train_perceptual"] = loss_perc.item()

        loss_g.backward()
        opt_g.step()
        logs["train_g"] = loss_g.item()
        return logs

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        fake = self.generator(batch["sketch"])
        return {"val_l1": self.l1_loss(fake, batch["color"]).item()}

    @torch.no_grad()
    def generate(self, sketch: torch.Tensor) -> torch.Tensor:
        """Colorize a batch of sketches (eval mode, no grad)."""
        self.generator.eval()
        return self.generator(sketch.to(self.device)).cpu()

    def _wandb_log_images(self) -> dict:
        """Log sketch | prediction | ground truth grid to W&B."""
        if self._wandb is None:
            return {}
        self.generator.eval()
        batch = next(iter(self.val_loader))
        batch = self._batch_to_device(batch)
        sketch = batch["sketch"][:4]
        color  = batch["color"][:4]
        with torch.no_grad():
            fake = self.generator(sketch)

        def to_grid(t: torch.Tensor):
            grid = make_grid(t.cpu() * 0.5 + 0.5, nrow=4, padding=2).clamp(0, 1)
            return self._wandb.Image(grid.permute(1, 2, 0).numpy())

        return {
            "val/sketch":       to_grid(sketch),
            "val/prediction":   to_grid(fake),
            "val/ground_truth": to_grid(color),
        }
