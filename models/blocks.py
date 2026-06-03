"""
Reusable building blocks for WindGapGAN models.

All blocks are designed to be composable and configurable.
No magic numbers — all hyperparameters are exposed as arguments.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolution → Normalization → Activation block.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        kernel_size: Convolution kernel size.
        padding: Padding size.
        use_batch_norm: Whether to use BatchNorm.
        activation: Activation function ('relu' or 'leaky_relu').
        dropout: Dropout probability (0 = disabled).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        use_batch_norm: bool = True,
        activation: str = "relu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=not use_batch_norm),
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))

        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "leaky_relu":
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        elif activation == "none":
            pass
        else:
            raise ValueError(f"Unknown activation: {activation}")

        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DoubleConvBlock(nn.Module):
    """
    Two consecutive ConvBlocks (standard U-Net building block).

    Conv → BN → ReLU → Conv → BN → ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, use_batch_norm=use_batch_norm, dropout=dropout),
            ConvBlock(out_channels, out_channels, use_batch_norm=use_batch_norm, dropout=0.0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.

    Conv → BN → ReLU → Conv → BN + skip → ReLU
    """

    def __init__(
        self,
        channels: int,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, use_batch_norm=use_batch_norm, dropout=dropout)
        self.conv2 = ConvBlock(channels, channels, use_batch_norm=use_batch_norm, activation="none")
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + residual
        return self.relu(out)


class DownBlock(nn.Module):
    """
    Encoder block: DoubleConv → MaxPool.

    Returns both the feature map (for skip connections) and the
    downsampled output.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv = DoubleConvBlock(in_channels, out_channels, use_batch_norm=use_batch_norm, dropout=dropout)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            features: Before pooling (for skip connections).
            pooled: After pooling (for next encoder stage).
        """
        features = self.conv(x)
        pooled = self.pool(features)
        return features, pooled


class UpBlock(nn.Module):
    """
    Decoder block: Upsample → Concatenate skip → DoubleConv.

    Uses bilinear upsampling followed by a 1×1 conv to match channels,
    which is more memory-efficient than transposed convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
        )
        self.conv = DoubleConvBlock(
            in_channels + skip_channels,
            out_channels,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Upsampled features from decoder.
            skip: Skip connection from encoder.
        """
        x = self.up(x)

        # Handle size mismatch due to odd dimensions
        if x.shape != skip.shape:
            diff_h = skip.shape[2] - x.shape[2]
            diff_w = skip.shape[3] - x.shape[3]
            x = nn.functional.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)
