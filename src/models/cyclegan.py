"""ResNet-based generator for CycleGAN (Zhu et al., 2017).

CycleGAN uses two generators (sketch->color, color->sketch) and two
PatchGAN discriminators (see patchgan.py, in_channels=3).
"""

import torch
import torch.nn as nn


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """ResNet generator with 9 residual blocks (256x256 resolution).

    Architecture: c7s1-64, d128, d256, 9×R256, u128, u64, c7s1-3 + Tanh.
    Reflection padding is used to reduce boundary artifacts.

    Args:
        in_channels: input channels (3).
        out_channels: output channels (3).
        n_blocks: number of residual blocks (9 for 256x256).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 n_blocks: int = 9):
        super().__init__()

        # Initial convolution block — c7s1-64
        encoder = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, 64, kernel_size=7, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]

        # Downsampling — d128, d256
        for in_ch, out_ch in [(64, 128), (128, 256)]:
            encoder += [
                nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]

        # Residual blocks — 9× R256
        residuals = [_ResidualBlock(256) for _ in range(n_blocks)]

        # Upsampling — u128, u64
        decoder = []
        for in_ch, out_ch in [(256, 128), (128, 64)]:
            decoder += [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2,
                                   padding=1, output_padding=1, bias=False),
                nn.InstanceNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]

        # Output convolution — c7s1-3
        decoder += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*encoder, *residuals, *decoder)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
