"""
PatchGAN Discriminator for WindGapGAN.

Conditional PatchGAN discriminator that classifies overlapping 70×70
patches as real or fake.  Standard architecture from Pix2Pix / SRGAN.

Input:
    Concatenation of [condition, image]:
        condition = masked_input (field * mask)   →  (B, 1, H, W)
        image     = prediction  OR  ground_truth  →  (B, 1, H, W)
    Total input channels: 2 (condition + image)

    Optionally includes the mask channel:
        condition = [field * mask, mask]           →  (B, 2, H, W)
        image     = prediction OR ground_truth     →  (B, 1, H, W)
    Total input channels: 3

Output:
    Patch-wise real/fake scores: (B, 1, H', W')

Architecture:
    4 downsampling ConvBlocks with increasing features.
    Spectral normalization on all conv layers for training stability.
    LeakyReLU(0.2) activation throughout.
    No BatchNorm on the first layer (standard PatchGAN practice).

Feature extraction:
    Intermediate activations can be returned for perceptual
    (feature-matching) loss — critical for learning textures.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def _apply_spectral_norm(module: nn.Module) -> nn.Module:
    """Apply spectral normalization to Conv2d layers in a module."""
    if isinstance(module, nn.Conv2d):
        return nn.utils.spectral_norm(module)
    return module


class DiscriminatorBlock(nn.Module):
    """
    Single discriminator downsampling block.

    Conv → [BN] → LeakyReLU

    Args:
        in_channels: Input channels.
        out_channels: Output channels.
        stride: Convolution stride (2 for downsampling, 1 for final).
        use_batch_norm: Whether to use BatchNorm (skip for first layer).
        use_spectral_norm: Whether to apply spectral normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 2,
        use_batch_norm: bool = True,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()

        conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=4, stride=stride, padding=1,
            bias=not use_batch_norm,
        )
        if use_spectral_norm:
            conv = nn.utils.spectral_norm(conv)

        layers: list[nn.Module] = [conv]
        if use_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class PatchGANDiscriminator(nn.Module):
    """
    Conditional PatchGAN discriminator (70×70 receptive field).

    Takes concatenated [condition, image] as input and produces
    a patch-wise real/fake probability map.

    Can return intermediate features for perceptual loss.

    Args:
        in_channels: Total input channels (condition + image).
            Default 3: [masked_field(1) + mask(1) + prediction/gt(1)]
        base_features: Base feature count (doubled at each stage).
        n_layers: Number of downsampling layers (default 3 → 70×70 RF).
        use_spectral_norm: Apply spectral normalization for stability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_features: int = 64,
        n_layers: int = 3,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()

        self.n_layers = n_layers

        # Layer 0: No BatchNorm on first layer (PatchGAN standard)
        self.layers = nn.ModuleList()
        self.layers.append(
            DiscriminatorBlock(
                in_channels, base_features,
                stride=2, use_batch_norm=False,
                use_spectral_norm=use_spectral_norm,
            )
        )

        # Layers 1 to n_layers-1: Downsampling with BatchNorm
        ch_in = base_features
        for i in range(1, n_layers):
            ch_out = min(base_features * (2 ** i), 512)
            self.layers.append(
                DiscriminatorBlock(
                    ch_in, ch_out,
                    stride=2, use_batch_norm=True,
                    use_spectral_norm=use_spectral_norm,
                )
            )
            ch_in = ch_out

        # Layer n_layers: stride=1 (no downsampling)
        ch_out = min(base_features * (2 ** n_layers), 512)
        self.layers.append(
            DiscriminatorBlock(
                ch_in, ch_out,
                stride=1, use_batch_norm=True,
                use_spectral_norm=use_spectral_norm,
            )
        )
        ch_in = ch_out

        # Final 1×1 conv → single-channel output (real/fake score per patch)
        final_conv = nn.Conv2d(ch_in, 1, kernel_size=4, stride=1, padding=1)
        if use_spectral_norm:
            final_conv = nn.utils.spectral_norm(final_conv)
        self.final = final_conv

        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            "PatchGANDiscriminator: in_ch=%d, base=%d, n_layers=%d, "
            "spectral_norm=%s, params=%s",
            in_channels, base_features, n_layers,
            use_spectral_norm, f"{total_params:,}",
        )

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> tuple[torch.Tensor, list[torch.Tensor]] | torch.Tensor:
        """
        Forward pass.

        Args:
            x: Concatenated input (B, C_cond + C_img, H, W).
            return_features: If True, also return intermediate features
                for perceptual (feature-matching) loss.

        Returns:
            If return_features=False:
                Patch scores: (B, 1, H', W')
            If return_features=True:
                (patch_scores, [feat_1, feat_2, ..., feat_n])
        """
        features = []
        h = x
        for layer in self.layers:
            h = layer(h)
            if return_features:
                features.append(h)

        out = self.final(h)

        if return_features:
            return out, features
        return out

    @staticmethod
    def build_input(
        masked_field: torch.Tensor,
        mask: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build discriminator input by concatenating condition and image.

        Args:
            masked_field: (B, 1, H, W) — observed field (zeros in gaps).
            mask: (B, 1, H, W) — observation mask (1=observed, 0=gap).
            image: (B, 1, H, W) — prediction or ground truth.

        Returns:
            (B, 3, H, W) — concatenated input.
        """
        return torch.cat([masked_field, mask, image], dim=1)
