"""Conditional VAE for sketch colorization (variant 5).

Conditions on the sketch image; the latent code captures color/style
variation, enabling diverse colorizations of the same sketch.
"""

import torch
import torch.nn as nn


class ConditionalVAE(nn.Module):
    """CVAE conditioned on the sketch.

    Encoder sees (color, sketch) and produces q(z | color, sketch);
    decoder reconstructs the color image from (z, sketch features).

    Args:
        latent_dim: dimensionality of the latent code.
        base_filters: filters in the first conv block.
    """

    def __init__(self, latent_dim: int = 128, base_filters: int = 64):
        super().__init__()
        # TODO: sketch encoder, posterior encoder, decoder with sketch
        # feature injection (concat or skip connections)
        raise NotImplementedError

    def forward(self, sketch: torch.Tensor, color: torch.Tensor):
        # Returns (reconstruction, mu, logvar)
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, sketch: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample diverse colorizations of a sketch with z ~ N(0, I)."""
        raise NotImplementedError
