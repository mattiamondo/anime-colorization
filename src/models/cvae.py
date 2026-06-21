"""Conditional VAE for sketch colorization (variant 5).

Architecture:
- sketch_enc: U-Net-style encoder extracts multi-scale features and a bottleneck
- posterior_enc: encodes (sketch + color) -> (mu, logvar) during training
- z_proj: projects sampled z to the bottleneck spatial resolution
- decoder: U-Net-style decoder with sketch skip connections reconstructs color

Training uses the reparameterization trick with z ~ q(z|sketch, color).
Inference samples z ~ N(0, I), enabling diverse colorizations of the same sketch.
"""

import torch
import torch.nn as nn

from .unet import _down_block, _up_block


class ConditionalVAE(nn.Module):
    """CVAE conditioned on the sketch via U-Net skip connections.

    Args:
        latent_dim: dimensionality of the latent code z.
        base_filters: filters in the first encoder block (default 64).
    """

    def __init__(self, latent_dim: int = 128, base_filters: int = 64,
                 use_colorgram: bool = True):
        super().__init__()
        f = base_filters
        self.latent_dim = latent_dim
        self.use_colorgram = use_colorgram

        # Sketch encoder: same blocks as UNet, provides skips + bottleneck.
        # s[0..6]: multi-scale features; s[7]=bottleneck is kept separate.
        self.sketch_enc = nn.ModuleList([
            _down_block(3, f,     normalize=False),  # 256->128  (B, 64, 128, 128)
            _down_block(f,   f*2),                   # 128->64   (B, 128, 64, 64)
            _down_block(f*2, f*4),                   # 64->32    (B, 256, 32, 32)
            _down_block(f*4, f*8),                   # 32->16    (B, 512, 16, 16)
            _down_block(f*8, f*8),                   # 16->8     (B, 512, 8, 8)
            _down_block(f*8, f*8),                   # 8->4      (B, 512, 4, 4)
            _down_block(f*8, f*8),                   # 4->2      (B, 512, 2, 2)
            _down_block(f*8, f*8, normalize=False),  # 2->1      (B, 512, 1, 1)
        ])

        # Posterior encoder q(z | sketch, color): 6-channel input (cat).
        # Global average pool collapses spatial dims before the FC layers.
        self.posterior_enc = nn.Sequential(
            _down_block(6,   f,   normalize=False),  # 256->128
            _down_block(f,   f*2),                   # 128->64
            _down_block(f*2, f*4),                   # 64->32
            _down_block(f*4, f*8),                   # 32->16
            _down_block(f*8, f*8),                   # 16->8
            nn.AdaptiveAvgPool2d(1),                 # (B, 512, 1, 1)
            nn.Flatten(),                            # (B, 512)
        )
        self.fc_mu     = nn.Linear(f*8, latent_dim)
        self.fc_logvar = nn.Linear(f*8, latent_dim)

        # Colorgram encoder: (16, 3) palette → latent_dim embedding.
        # Summed with z before z_proj so guided and free modes share the same path.
        # 16 cells × 3 RGB channels = 48 input features.
        if use_colorgram:
            self.colorgram_enc = nn.Sequential(
                nn.Linear(48, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, latent_dim),
            )

        # Project z (optionally + colorgram_emb) to bottleneck shape for additive injection.
        self.z_proj = nn.Sequential(
            nn.Linear(latent_dim, f*8),
            nn.ReLU(inplace=True),
        )

        # Decoder: identical channel progression to the UNet decoder.
        # The first up_block receives the z-injected bottleneck (512 ch).
        # Each subsequent block receives the upsampled output concatenated
        # with the matching sketch skip (hence the f*16 = 2*f*8 inputs).
        self.ups = nn.ModuleList([
            _up_block(f*8,  f*8,  dropout=True),   # 1->2   (B, 512)
            _up_block(f*16, f*8,  dropout=True),   # 2->4   cat s[6] (512)
            _up_block(f*16, f*8,  dropout=True),   # 4->8   cat s[5] (512)
            _up_block(f*16, f*8),                   # 8->16  cat s[4] (512)
            _up_block(f*16, f*4),                   # 16->32 cat s[3] (512)
            _up_block(f*8,  f*2),                   # 32->64 cat s[2] (256)
            _up_block(f*4,  f),                     # 64->128 cat s[1](128)
        ])
        self.final = nn.Sequential(
            nn.ConvTranspose2d(f*2, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    # Internal helpers

    def _encode_sketch(
        self, sketch: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        """Run the sketch encoder.

        Returns:
            skips: list of 7 intermediate feature maps (deepest first after
                reversal — ready for the decoder).
            bottleneck: (B, 512, 1, 1) feature from the last down block.
        """
        x = sketch
        skips: list[torch.Tensor] = []
        for block in self.sketch_enc[:-1]:
            x = block(x)
            skips.append(x)
        bottleneck = self.sketch_enc[-1](x)
        skips = skips[::-1]  # deepest first, matching decoder order
        return skips, bottleneck

    def _decode(
        self, x: torch.Tensor, skips: list[torch.Tensor]
    ) -> torch.Tensor:
        """Run the decoder given a bottleneck tensor and sketch skip features."""
        for i, up in enumerate(self.ups):
            x = up(x)
            x = torch.cat([x, skips[i]], dim=1)
        return self.final(x)

    def _posterior(
        self, sketch: torch.Tensor, color: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.posterior_enc(torch.cat([sketch, color], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def _reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def _colorgram_embedding(
        self, colorgram: torch.Tensor | None, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        """Return (B, latent_dim) colorgram embedding, or zeros if not available."""
        if colorgram is not None and self.use_colorgram:
            # colorgram: (B, 16, 3) → flatten → (B, 48)
            return self.colorgram_enc(colorgram.view(batch_size, -1).to(device))
        return torch.zeros(batch_size, self.latent_dim, device=device)

    # Public API

    def forward(
        self, sketch: torch.Tensor, color: torch.Tensor,
        colorgram: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Training forward pass.

        Args:
            sketch: (B, 3, H, W) grayscale sketch in [-1, 1].
            color: (B, 3, H, W) color target in [-1, 1].
            colorgram: optional (B, 16, 3) palette in [-1, 1].

        Returns:
            reconstruction: (B, 3, H, W) in [-1, 1]
            mu: (B, latent_dim)
            logvar: (B, latent_dim)
        """
        skips, _ = self._encode_sketch(sketch)
        mu, logvar = self._posterior(sketch, color)
        z = self._reparameterize(mu, logvar)
        z = z + self._colorgram_embedding(colorgram, z.size(0), z.device)
        z_feat = self.z_proj(z).view(z.size(0), -1, 1, 1)
        reconstruction = self._decode(z_feat, skips)
        return reconstruction, mu, logvar

    @torch.no_grad()
    def sample(
        self, sketch: torch.Tensor, n_samples: int = 1,
        colorgram: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample diverse colorizations of a sketch by drawing z ~ N(0, I).

        Args:
            sketch: (B, 3, H, W) input sketch batch.
            n_samples: number of independent colorizations to generate.
            colorgram: optional (B, 16, 3) palette in [-1, 1]. When provided,
                all samples are guided by the same palette (guided mode).
                When None, samples are drawn freely from the prior (free mode).

        Returns:
            Tensor of shape (n_samples, B, 3, H, W).
        """
        skips, _ = self._encode_sketch(sketch)
        cg_emb = self._colorgram_embedding(colorgram, sketch.size(0), sketch.device)
        results = []
        for _ in range(n_samples):
            z = torch.randn(sketch.size(0), self.latent_dim, device=sketch.device)
            z_feat = self.z_proj(z + cg_emb).view(z.size(0), -1, 1, 1)
            # Clone skips so each sample gets its own copy (no in-place aliasing).
            colorization = self._decode(
                z_feat,
                [s.clone() for s in skips],
            )
            results.append(colorization)
        return torch.stack(results)  # (n_samples, B, 3, H, W)
