"""U-Net generator with skip connections."""

import torch
import torch.nn as nn


def _down_block(in_ch: int, out_ch: int, normalize: bool = True) -> nn.Sequential:
    """Conv 4x4 stride 2 -> [InstanceNorm] -> LeakyReLU. Halves H and W."""
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=not normalize)
    ]
    if normalize:
        layers.append(nn.InstanceNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return nn.Sequential(*layers)


def _up_block(in_ch: int, out_ch: int, dropout: bool = False) -> nn.Sequential:
    """TransposedConv 4x4 stride 2 -> InstanceNorm -> [Dropout] -> ReLU. Doubles H and W."""
    layers: list[nn.Module] = [
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
        nn.InstanceNorm2d(out_ch),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class UNetGenerator(nn.Module):
    """
    Args:
        in_channels: input channels (3 for RGB sketch).
        out_channels: output channels (3 for RGB color image).
        base_filters: filters in the first encoder block (default 64).
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3,
                 base_filters: int = 64):
        super().__init__()
        f = base_filters
        self.downs = nn.ModuleList([
            _down_block(in_channels, f, normalize=False),  # 256 -> 128
            _down_block(f, f * 2),                         # 128 -> 64
            _down_block(f * 2, f * 4),                     # 64 -> 32
            _down_block(f * 4, f * 8),                     # 32 -> 16
            _down_block(f * 8, f * 8),                     # 16 -> 8
            _down_block(f * 8, f * 8),                     # 8 -> 4
            _down_block(f * 8, f * 8),                     # 4 -> 2
            _down_block(f * 8, f * 8, normalize=False),    # 2 -> 1 (bottleneck)
        ])
        self.ups = nn.ModuleList([
            _up_block(f * 8, f * 8, dropout=True),         # 1 -> 2
            _up_block(f * 16, f * 8, dropout=True),        # 2 -> 4
            _up_block(f * 16, f * 8, dropout=True),        # 4 -> 8
            _up_block(f * 16, f * 8),                      # 8 -> 16
            _up_block(f * 16, f * 4),                      # 16 -> 32
            _up_block(f * 8, f * 2),                       # 32 -> 64
            _up_block(f * 4, f),                           # 64 -> 128
        ])
        self.final = nn.Sequential(
            nn.ConvTranspose2d(f * 2, out_channels, kernel_size=4, stride=2,
                               padding=1),                 # 128 -> 256
            nn.Tanh(),                                     # images in [-1, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
        skips = skips[:-1][::-1]  # inverted encoder outputs, no bottleneck
        for i, up in enumerate(self.ups):
            x = up(x)
            x = torch.cat([x, skips[i]], dim=1)
        return self.final(x)
