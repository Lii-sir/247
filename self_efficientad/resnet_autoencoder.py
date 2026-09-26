"""Channel-scaled EfficientAD-style autoencoder for native ResNet features."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class ResNetFeatureAutoEncoder(nn.Module):
    """Compress RGB, then reconstruct C teacher features at their native size.

    Retains the six-convolution encoder and eight-convolution decoder pattern,
    dropout and absence of skip connections. The decoder never builds a
    PDN-sized feature map merely to downsample it back to the ResNet grid.
    """

    def __init__(self, out_channels: int) -> None:
        super().__init__()
        from .torch_model import Encoder

        self.out_channels = out_channels
        self.hidden_channels = min(256, max(64, out_channels // 2))
        self.encoder = Encoder(self.hidden_channels // 2, self.hidden_channels)
        self.decoder = nn.ModuleList(
            [nn.Conv2d(self.hidden_channels, self.hidden_channels, 4, padding=2)
             for _ in range(6)]
            + [nn.Conv2d(self.hidden_channels, self.hidden_channels, 3, padding=1),
               nn.Conv2d(self.hidden_channels, out_channels, 3, padding=1)]
        )
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor, feature_size: tuple[int, int] | torch.Size) -> torch.Tensor:
        from .torch_model import imagenet_norm_batch

        height, width = feature_size
        if height < 2 or width < 2:
            raise ValueError("ResNet AE feature_size 的高和宽必须至少为 2。")
        x = self.encoder(imagenet_norm_batch(x))
        sizes = [
            (max(2, math.ceil(height / 4)), max(2, math.ceil(width / 4))),
            (max(2, math.ceil(height / 2)), max(2, math.ceil(width / 2))),
            *((height, width),) * 4,
        ]
        for conv, (h, w) in zip(self.decoder[:6], sizes, strict=True):
            # A 4x4 convolution with padding=2 grows each dimension by one.
            x = F.interpolate(x, size=(h - 1, w - 1), mode="bilinear", align_corners=False)
            x = self.dropout(F.relu(conv(x)))
        x = F.relu(self.decoder[6](x))
        # Standardized teacher targets can be negative: no output activation.
        return self.decoder[7](x)
