"""Trainer for CycleGAN (variant 4, unpaired).

IMPORTANT: must never use the pairing information — sketch and color
batches are sampled independently (dataset with paired=False). The pairs
are only used at EVALUATION time to compute SSIM/PSNR against ground truth.
"""

import random

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from ..losses import GANLoss
from ..models import PatchGANDiscriminator, ResnetGenerator
from .base import BaseTrainer


class _ImageBuffer:
    """Replay buffer to stabilize discriminator training (Shrivastava et al., 2017).

    Stores up to `size` generated images. When queried:
    - with prob 0.5: returns the incoming image and stores it in the buffer.
    - with prob 0.5: returns a random image from the buffer, replacing it
      with the incoming one.
    """

    def __init__(self, size: int = 50):
        self.size = size
        self.buffer: list[torch.Tensor] = []

    def query(self, images: torch.Tensor) -> torch.Tensor:
        if self.size == 0:
            return images
        results = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.buffer) < self.size:
                self.buffer.append(image)
                results.append(image)
            elif random.random() < 0.5:
                idx = random.randrange(len(self.buffer))
                stored = self.buffer[idx].clone()
                self.buffer[idx] = image
                results.append(stored)
            else:
                results.append(image)
        return torch.cat(results, dim=0)


class CycleGANTrainer(BaseTrainer):
    """Two ResNet generators + two PatchGAN discriminators.

    Losses: LSGAN adversarial, cycle-consistency (lambda_cycle=10),
    identity loss (lambda_identity=0.5 * lambda_cycle).

    The train dataloader must use paired=False. The val dataloader uses
    paired=True so we can monitor val_l1 against ground truth for
    checkpoint selection.

    LR schedule: constant for the first `decay_start` epochs, then linear
    decay to zero over the remaining epochs (paper: 100+100).

    Args:
        lambda_cycle: weight of cycle-consistency loss (paper: 10).
        lambda_identity: weight of identity loss (paper: 5 = 0.5 * lambda_cycle).
        lr: initial learning rate for both optimizers (paper: 2e-4).
        betas: Adam betas.
        decay_start: epoch at which LR decay begins (paper: 100).
        use_amp: enable automatic mixed precision (fp16 autocast + GradScaler).
            CycleGAN is compute-bound (6 ResNet generator forwards + 2
            discriminators per step), so AMP roughly halves the step time on
            the RTX 3090. Disabled automatically on non-CUDA devices.
    """

    def __init__(self, *, lambda_cycle: float = 10.0,
                 lambda_identity: float = 5.0, lr: float = 2e-4,
                 betas: tuple[float, float] = (0.5, 0.999),
                 decay_start: int = 100, use_amp: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.decay_start = decay_start
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = GradScaler(self.device.type, enabled=self.use_amp)

        # Generators
        self.G_s2c = self.register_network("G_s2c", ResnetGenerator())
        self.G_c2s = self.register_network("G_c2s", ResnetGenerator())

        # Discriminators (unconditional: single image, in_channels=3)
        self.D_color = self.register_network(
            "D_color", PatchGANDiscriminator(in_channels=3))
        self.D_sketch = self.register_network(
            "D_sketch", PatchGANDiscriminator(in_channels=3))

        # Replay buffers
        self.buffer_color = _ImageBuffer(50)
        self.buffer_sketch = _ImageBuffer(50)

        # Losses
        self.gan_loss = GANLoss("lsgan").to(self.device)
        self.l1_loss = nn.L1Loss()

        # Optimizers: one joint optimizer for both generators (paper convention)
        self.optimizers["G"] = torch.optim.Adam(
            list(self.G_s2c.parameters()) + list(self.G_c2s.parameters()),
            lr=lr, betas=betas)
        self.optimizers["D_color"] = torch.optim.Adam(
            self.D_color.parameters(), lr=lr, betas=betas)
        self.optimizers["D_sketch"] = torch.optim.Adam(
            self.D_sketch.parameters(), lr=lr, betas=betas)

        # LR schedulers (built after optimizers are ready)
        self._schedulers: dict[str, torch.optim.lr_scheduler.LRScheduler] = {}

    def _build_schedulers(self, total_epochs: int) -> None:
        """Call once before the training loop starts (fit() does this)."""
        decay_epochs = max(total_epochs - self.decay_start, 1)

        def lr_lambda(epoch: int) -> float:
            if epoch < self.decay_start:
                return 1.0
            return max(0.0, 1.0 - (epoch - self.decay_start) / decay_epochs)

        for key, opt in self.optimizers.items():
            self._schedulers[key] = torch.optim.lr_scheduler.LambdaLR(
                opt, lr_lambda=lr_lambda, last_epoch=self.epoch - 1)

    def fit(self, epochs: int) -> dict[str, list[float]]:
        self._build_schedulers(epochs)
        for epoch in range(self.epoch + 1, epochs + 1):
            self.epoch = epoch

            self._set_train_mode(True)
            train_logs = self._run_epoch(
                self.train_loader, self.training_step,
                desc=f"epoch {epoch}/{epochs} [train]")

            self._set_train_mode(False)
            with torch.no_grad():
                val_logs = self._run_epoch(
                    self.val_loader, self.validation_step,
                    desc=f"epoch {epoch}/{epochs} [val]")

            for sched in self._schedulers.values():
                sched.step()

            logs = {**train_logs, **val_logs}
            for key, value in logs.items():
                self.history.setdefault(key, []).append(value)

            self.save_checkpoint("last.pt")
            monitored = logs.get(self.monitor)
            if monitored is not None and monitored < self.best_metric:
                self.best_metric = monitored
                self.save_checkpoint("best.pt")

            summary = " | ".join(f"{k}={v:.4f}" for k, v in logs.items())
            print(f"epoch {epoch:3d}/{epochs} | {summary}")

        return self.history

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        sketch, color = batch["sketch"], batch["color"]

        # --- Generator step (freeze discriminators) ---
        for p in list(self.D_color.parameters()) + list(self.D_sketch.parameters()):
            p.requires_grad_(False)

        opt_g = self.optimizers["G"]
        opt_g.zero_grad(set_to_none=True)

        with autocast(self.device.type, enabled=self.use_amp):
            # Forward pass
            fake_color  = self.G_s2c(sketch)        # G_s2c: sketch -> color
            fake_sketch = self.G_c2s(color)         # G_c2s: color  -> sketch
            rec_sketch  = self.G_c2s(fake_color)    # cycle forward:  sketch -> color -> sketch
            rec_color   = self.G_s2c(fake_sketch)   # cycle backward: color  -> sketch -> color
            idt_color   = self.G_s2c(color)         # identity: G_s2c should preserve color images
            idt_sketch  = self.G_c2s(sketch)        # identity: G_c2s should preserve sketch images

            loss_adv = (self.gan_loss(self.D_color(fake_color), True)
                        + self.gan_loss(self.D_sketch(fake_sketch), True))
            loss_cycle = (self.l1_loss(rec_sketch, sketch)
                          + self.l1_loss(rec_color, color))
            loss_idt = (self.l1_loss(idt_color, color)
                        + self.l1_loss(idt_sketch, sketch))

            loss_g = (loss_adv
                      + self.lambda_cycle * loss_cycle
                      + self.lambda_identity * loss_idt)

        self.scaler.scale(loss_g).backward()
        self.scaler.step(opt_g)

        # --- D_color step ---
        for p in self.D_color.parameters():
            p.requires_grad_(True)

        opt_dc = self.optimizers["D_color"]
        opt_dc.zero_grad(set_to_none=True)
        buffered_fake_color = self.buffer_color.query(fake_color.detach())
        with autocast(self.device.type, enabled=self.use_amp):
            loss_dc = 0.5 * (self.gan_loss(self.D_color(color), True)
                             + self.gan_loss(self.D_color(buffered_fake_color), False))
        self.scaler.scale(loss_dc).backward()
        self.scaler.step(opt_dc)

        # --- D_sketch step ---
        for p in self.D_sketch.parameters():
            p.requires_grad_(True)

        opt_ds = self.optimizers["D_sketch"]
        opt_ds.zero_grad(set_to_none=True)
        buffered_fake_sketch = self.buffer_sketch.query(fake_sketch.detach())
        with autocast(self.device.type, enabled=self.use_amp):
            loss_ds = 0.5 * (self.gan_loss(self.D_sketch(sketch), True)
                             + self.gan_loss(self.D_sketch(buffered_fake_sketch), False))
        self.scaler.scale(loss_ds).backward()
        self.scaler.step(opt_ds)

        # One scaler update per iteration, after all optimizer steps.
        self.scaler.update()

        return {
            "train_G":       loss_g.item(),
            "train_D_color": loss_dc.item(),
            "train_D_sketch": loss_ds.item(),
            "train_cycle":   loss_cycle.item(),
            "train_idt":     loss_idt.item(),
        }

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        fake_color = self.G_s2c(batch["sketch"])
        return {"val_l1": self.l1_loss(fake_color, batch["color"]).item()}

    @torch.no_grad()
    def generate(self, sketch: torch.Tensor) -> torch.Tensor:
        """Colorize a batch of sketches (eval mode, no grad)."""
        self.G_s2c.eval()
        return self.G_s2c(sketch.to(self.device)).cpu()
