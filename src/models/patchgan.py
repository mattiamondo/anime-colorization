"""PatchGAN discriminator (70x70 receptive field).

Used by Pix2Pix (conditional: input = sketch concatenated with real/fake
color image) and CycleGAN (unconditional: input = single image).
"""

import torch
import torch.nn as nn


class PatchGANDiscriminator(nn.Module):
    """Standard 70x70 PatchGAN (Isola et al., 2017).

    C64-C128-C256-C512 followed by a 1-channel conv: each output logit
    judges a 70x70 patch of the input (30x30 map for 256x256 inputs).

    Args:
        in_channels: 6 for conditional Pix2Pix (sketch + image),
            3 for unconditional CycleGAN.
        base_filters: filters in the first conv block (default 64).
    """

    def __init__(self, in_channels: int = 6, base_filters: int = 64):
        super().__init__()
        f = base_filters

        def block(in_ch: int, out_ch: int, stride: int, normalize: bool = True):
            layers: list[nn.Module] = [
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=stride,
                        padding=1, bias=not normalize)
            ]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, f, stride=2, normalize=False),  # C64
            *block(f, f * 2, stride=2),                         # C128
            *block(f * 2, f * 4, stride=2),                     # C256
            *block(f * 4, f * 8, stride=1),                     # C512
            nn.Conv2d(f * 8, 1, kernel_size=4, stride=1, padding=1),  # patch logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
