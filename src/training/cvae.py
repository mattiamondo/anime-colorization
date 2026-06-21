"""Trainer for the Conditional VAE (variant 5).

Loss: ELBO = L1 reconstruction + beta * KL divergence.

KL annealing (beta warmup) avoids posterior collapse: beta starts at 0
and increases linearly to its target value over `beta_warmup_epochs` epochs,
giving the decoder time to learn to reconstruct before the KL term kicks in.

Early stopping is built in: training halts when `monitor` (default: val_recon)
has not improved for `patience` consecutive epochs.
"""

import torch
import torch.nn as nn
from torchvision.utils import make_grid

from ..models import ConditionalVAE
from .base import BaseTrainer


class _DeterministicWrapper(nn.Module):
    """Wraps ConditionalVAE for evaluate_model: sketch -> color with z=0."""

    def __init__(self, cvae: ConditionalVAE):
        super().__init__()
        self.cvae = cvae

    def forward(self, sketch: torch.Tensor) -> torch.Tensor:
        skips, _ = self.cvae._encode_sketch(sketch)
        z = torch.zeros(sketch.size(0), self.cvae.latent_dim, device=sketch.device)
        z_feat = self.cvae.z_proj(z).view(z.size(0), -1, 1, 1)
        return self.cvae._decode(z_feat, skips)


class CVAETrainer(BaseTrainer):
    """CVAE trainer with reconstruction + KL loss and early stopping.

    Args:
        latent_dim: size of the latent code (default 128).
        beta: target weight of the KL term (default 1.0).
        beta_warmup_epochs: epochs over which beta is linearly annealed from 0
            to its target value. Set to 0 to disable annealing.
        patience: early-stopping patience in epochs (default 15).
        lr: Adam learning rate (default 1e-4).
        betas: Adam beta coefficients.
    """

    def __init__(self, *, latent_dim: int = 128, beta: float = 1.0,
                 beta_warmup_epochs: int = 10, patience: int = 15,
                 early_stopping: bool = True,
                 lr: float = 1e-4, betas: tuple[float, float] = (0.5, 0.999),
                 use_colorgram: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.beta = beta
        self.beta_warmup_epochs = beta_warmup_epochs
        self.patience = patience
        self.early_stopping = early_stopping

        self.model = self.register_network(
            "cvae", ConditionalVAE(latent_dim=latent_dim, use_colorgram=use_colorgram))
        self.optimizers["cvae"] = torch.optim.Adam(
            self.model.parameters(), lr=lr, betas=betas)
        self.l1_loss = nn.L1Loss()

    @property
    def current_beta(self) -> float:
        """KL weight, linearly annealed from 0 -> beta over warmup epochs."""
        if self.beta_warmup_epochs <= 0:
            return self.beta
        return self.beta * min(1.0, self.epoch / self.beta_warmup_epochs)

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        sketch, color = batch["sketch"], batch["color"]
        colorgram = batch.get("colorgram")

        opt = self.optimizers["cvae"]
        opt.zero_grad(set_to_none=True)

        reconstruction, mu, logvar = self.model(sketch, color, colorgram)

        loss_recon = self.l1_loss(reconstruction, color)
        # KL divergence per element, averaged over the batch.
        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = loss_recon + self.current_beta * loss_kl
        loss.backward()
        opt.step()

        return {
            "train_recon": loss_recon.item(),
            "train_kl":    loss_kl.item(),
            "train_loss":  loss.item(),
        }

    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        sketch, color = batch["sketch"], batch["color"]
        colorgram = batch.get("colorgram")
        reconstruction, mu, logvar = self.model(sketch, color, colorgram)
        loss_recon = self.l1_loss(reconstruction, color)
        loss_kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return {
            "val_recon": loss_recon.item(),
            "val_kl":    loss_kl.item(),
            "val_loss":  (loss_recon + self.current_beta * loss_kl).item(),
        }

    def fit(self, epochs: int) -> dict[str, list[float]]:
        """Train with early stopping based on `self.monitor`."""
        no_improve = 0

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

            logs = {**train_logs, **val_logs}
            for key, value in logs.items():
                self.history.setdefault(key, []).append(value)

            self.save_checkpoint("last.pt")

            monitored = logs.get(self.monitor)
            if monitored is not None and monitored < self.best_metric:
                self.best_metric = monitored
                self.save_checkpoint("best.pt")
                no_improve = 0
            else:
                no_improve += 1

            summary = " | ".join(f"{k}={v:.4f}" for k, v in logs.items())
            print(f"epoch {epoch:3d}/{epochs} | beta={self.current_beta:.3f} | {summary}")

            if self._wandb is not None:
                self._wandb.log({**logs, "beta": self.current_beta}, step=self.epoch)
                if (self.wandb_log_images_every > 0
                        and self.epoch % self.wandb_log_images_every == 0):
                    img_logs = self._wandb_log_images()
                    if img_logs:
                        self._wandb.log(img_logs, step=self.epoch)

            if self.early_stopping and no_improve >= self.patience:
                print(f"Early stopping: no improvement in {self.patience} epochs.")
                break

        return self.history

    # Inference helpers

    @torch.no_grad()
    def generate(
        self, sketch: torch.Tensor,
        colorgram: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic colorization using z=0 (prior mean).

        Args:
            colorgram: optional (B, 16, 3) palette tensor for guided generation.
        """
        self.model.eval()
        sketch = sketch.to(self.device)
        cg = colorgram.to(self.device) if colorgram is not None else None
        skips, _ = self.model._encode_sketch(sketch)
        z = torch.zeros(sketch.size(0), self.model.latent_dim, device=self.device)
        z = z + self.model._colorgram_embedding(cg, z.size(0), self.device)
        z_feat = self.model.z_proj(z).view(z.size(0), -1, 1, 1)
        return self.model._decode(z_feat, skips).cpu()

    @torch.no_grad()
    def sample_diverse(
        self, sketch: torch.Tensor, n_samples: int = 6,
        colorgram: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return n_samples diverse colorizations, shape (n_samples, B, 3, H, W).

        Args:
            colorgram: optional (B, 16, 3) palette for guided mode. When None,
                samples freely from the prior (free/stochastic mode).
        """
        self.model.eval()
        cg = colorgram.to(self.device) if colorgram is not None else None
        return self.model.sample(sketch.to(self.device), n_samples, cg).cpu()

    def _wandb_log_images(self) -> dict:
        """Log a qualitative grid: sketch | free sample | guided sample | GT."""
        if self._wandb is None:
            return {}
        self.model.eval()
        # Grab a fixed small batch from the val loader (first batch, up to 4 images).
        batch = next(iter(self.val_loader))
        batch = self._batch_to_device(batch)
        sketch  = batch["sketch"][:4]
        color   = batch["color"][:4]
        cg      = batch.get("colorgram")
        if cg is not None:
            cg = cg[:4]

        with torch.no_grad():
            free   = self.model.sample(sketch, n_samples=1)[0]          # (B, 3, H, W)
            guided = self.model.sample(sketch, n_samples=1, colorgram=cg)[0]

        def to_grid(t: torch.Tensor):
            grid = make_grid(t.cpu() * 0.5 + 0.5, nrow=4, padding=2).clamp(0, 1)
            return self._wandb.Image(grid.permute(1, 2, 0).numpy())

        return {
            "val/sketch":       to_grid(sketch),
            "val/free":         to_grid(free),
            "val/guided":       to_grid(guided),
            "val/ground_truth": to_grid(color),
        }

    @property
    def generator(self) -> nn.Module:
        """Expose a sketch->color module compatible with evaluate_model."""
        return _DeterministicWrapper(self.model)
