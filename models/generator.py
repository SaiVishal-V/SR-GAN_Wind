"""
SRResNet Generator for wind-speed super-resolution.

Architecture (per project plan + review changes):
    Conv → N Residual Blocks → Global Skip → 2× PixelShuffle → Final Conv

Key design decisions:
    - NO BatchNorm (review #1: harmful for scientific continuous fields)
    - Configurable residual blocks (review #13: default 8)
    - 1-channel input/output (wind speed, NOT RGB)
    - PReLU activation
    - PixelShuffle upsampling (2×2 = 4× total)
"""

import torch
import torch.nn as nn
from typing import Optional


class ResidualBlock(nn.Module):
    """
    Residual block WITHOUT BatchNorm.

    Architecture:
        Conv2D → PReLU → Conv2D → + skip
    """

    def __init__(self, num_features: int = 64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
            nn.PReLU(num_features),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class UpsampleBlock(nn.Module):
    """
    PixelShuffle upsampling block (2× spatial resolution).

    Architecture:
        Conv2D → PixelShuffle(2) → PReLU
    """

    def __init__(self, num_features: int = 64, scale: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features * scale * scale, kernel_size=3, padding=1),
            nn.PixelShuffle(scale),
            nn.PReLU(num_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SRResNet(nn.Module):
    """
    SRResNet Generator for 4× wind-speed super-resolution.

    Architecture:
        Initial Conv(9×9) + PReLU
        → N × ResidualBlock (no BN)
        → Conv(3×3) + global skip
        → 2 × PixelShuffle(2×) blocks
        → Final Conv(9×9)

    Args:
        in_channels: Number of input channels (1 for wind speed).
        num_features: Number of feature maps in residual blocks.
        num_residual_blocks: Number of residual blocks.
        scale_factor: Super-resolution scale factor (must be 4).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_features: int = 64,
        num_residual_blocks: int = 8,
        scale_factor: int = 4,
    ):
        super().__init__()
        assert scale_factor == 4, "Only 4× scale factor is supported"

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_features, kernel_size=9, padding=4),
            nn.PReLU(num_features),
        )

        # Residual blocks
        res_blocks = [ResidualBlock(num_features) for _ in range(num_residual_blocks)]
        self.residual_blocks = nn.Sequential(*res_blocks)

        # Post-residual conv (before global skip)
        self.post_residual = nn.Conv2d(
            num_features, num_features, kernel_size=3, padding=1
        )

        # Upsampling: 2 × PixelShuffle(2) = 4× total
        self.upsample = nn.Sequential(
            UpsampleBlock(num_features, scale=2),
            UpsampleBlock(num_features, scale=2),
        )

        # Final reconstruction
        self.final = nn.Conv2d(num_features, in_channels, kernel_size=9, padding=4)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Kaiming initialization for conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: LR input tensor of shape (B, 1, H, W).

        Returns:
            SR output tensor of shape (B, 1, 4H, 4W).
        """
        initial = self.initial(x)
        residual = self.post_residual(self.residual_blocks(initial))
        features = initial + residual  # Global skip connection
        upsampled = self.upsample(features)
        out = self.final(upsampled)
        return out
