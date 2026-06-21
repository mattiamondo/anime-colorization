"""Global discriminator: classifies the entire image with a single scalar.

Used by variant 6 (Pix2Pix + global discriminator) to contrast the
patch-level adversarial signal of PatchGAN with a global real/fake judgment.
"""

import torch
import torch.nn as nn


class GlobalDiscriminator(nn.Module):
    """Global discriminator (single scalar output per image).

    Same C64-C128-C256-C512 backbone as PatchGAN but the C512 block uses
    stride=2 (not 1), followed by global average pooling and a linear head,
    so the whole image is judged with one logit instead of a spatial patch map.

    Args:
        in_channels: 6 for conditional Pix2Pix (sketch + image), 3 otherwise.
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

        self.features = nn.Sequential(
            *block(in_channels, f,     stride=2, normalize=False),  # 256 → 128
            *block(f,     f * 2, stride=2),                          # 128 → 64
            *block(f * 2, f * 4, stride=2),                          # 64  → 32
            *block(f * 4, f * 8, stride=2),                          # 32  → 16 (stride=2, vs 1 in PatchGAN)
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(f * 8, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
