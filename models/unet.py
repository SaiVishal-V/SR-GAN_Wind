"""
Masked U-Net for gap filling (Phase 1 baseline).

Architecture:
    Input:  (B, T, 2, H, W)  — [field + mask] per timestep
    Output: (B, T, 1, H, W)  — reconstructed field

For Phase 1, each timestep is processed independently (no temporal
modeling). The temporal dimension is folded into the batch dimension:
    (B, T, 2, H, W) → (B*T, 2, H, W) → model → (B*T, 1, H, W) → unfold

Hard constraint: observed pixels are preserved exactly via masked merge:
    output = mask * input_field + (1 - mask) * predicted_field

Architecture:
    Encoder:    depth DownBlocks (base_features → base_features * 2^depth)
    Bottleneck: 2 ResidualBlocks
    Decoder:    depth UpBlocks with skip connections
    Head:       1×1 Conv → output channel
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from models.attention import SelfAttention2d
from models.blocks import DoubleConvBlock, DownBlock, ResidualBlock, UpBlock

logger = logging.getLogger(__name__)


class MaskedUNet(nn.Module):
    """
    U-Net adapted for gap filling with masked reconstruction.

    The model takes a wind field with gaps (filled with 0) and an
    observation mask as input, and predicts the complete field.
    Observed pixels are preserved exactly in the output.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_features: int = 32,
        depth: int = 4,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        use_attention: bool = True,
        use_tanh: bool = False,
        hard_merge: bool = True,
    ) -> None:
        """
        Args:
            in_channels: Input channels (field + mask = 2).
            out_channels: Output channels (reconstructed field = 1).
            base_features: Base number of feature maps.
            depth: Number of encoder/decoder stages.
            dropout: Dropout probability.
            use_batch_norm: Whether to use batch normalization.
            use_attention: Whether to use self-attention at bottleneck.
            use_tanh: Whether to apply Tanh to raw prediction (GAN mode).
            hard_merge: Whether to hard-merge observed pixels (set False
                for GAN training where softer gradients are preferred).
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.hard_merge = hard_merge

        # ── Encoder ────────────────────────────────────────────────────
        self.encoder_blocks = nn.ModuleList()
        ch_in = in_channels
        encoder_channels = []

        for i in range(depth):
            ch_out = base_features * (2 ** i)
            self.encoder_blocks.append(
                DownBlock(ch_in, ch_out, use_batch_norm=use_batch_norm, dropout=dropout if i >= 2 else 0.0)
            )
            encoder_channels.append(ch_out)
            ch_in = ch_out

        # ── Bottleneck ─────────────────────────────────────────────────
        bottleneck_ch = base_features * (2 ** depth)
        bottleneck_layers = [
            DoubleConvBlock(ch_in, bottleneck_ch, use_batch_norm=use_batch_norm, dropout=dropout),
            ResidualBlock(bottleneck_ch, use_batch_norm=use_batch_norm, dropout=dropout),
        ]
        if use_attention:
            bottleneck_layers.append(SelfAttention2d(bottleneck_ch))
        bottleneck_layers.append(
            ResidualBlock(bottleneck_ch, use_batch_norm=use_batch_norm, dropout=dropout),
        )
        self.bottleneck = nn.Sequential(*bottleneck_layers)

        # ── Decoder ────────────────────────────────────────────────────
        self.decoder_blocks = nn.ModuleList()
        ch_in = bottleneck_ch

        for i in range(depth - 1, -1, -1):
            ch_skip = encoder_channels[i]
            ch_out = encoder_channels[i]
            self.decoder_blocks.append(
                UpBlock(ch_in, ch_skip, ch_out, use_batch_norm=use_batch_norm, dropout=dropout if i >= 2 else 0.0)
            )
            ch_in = ch_out

        # ── Output Head ────────────────────────────────────────────────
        self.head = nn.Conv2d(ch_in, out_channels, kernel_size=1)
        self.use_tanh = use_tanh

        # Log model size
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "MaskedUNet: depth=%d, base_features=%d, params=%s (trainable=%s)",
            depth,
            base_features,
            f"{total_params:,}",
            f"{trainable_params:,}",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with optional temporal dimension handling.

        Args:
            x: Input tensor.
                If 5D: (B, T, C, H, W) — temporal sequence.
                If 4D: (B, C, H, W) — single frame.

        Returns:
            Reconstructed field with observed pixels preserved.
                If 5D input: (B, T, 1, H, W)
                If 4D input: (B, 1, H, W)
        """
        has_time = x.ndim == 5
        if has_time:
            B, T, C, H, W = x.shape
            # Fold time into batch: (B*T, C, H, W)
            x = x.reshape(B * T, C, H, W)
        else:
            B, C, H, W = x.shape
            T = 1

        # Extract mask channel (channel index 1)
        mask = x[:, 1:2, :, :]   # (B*T, 1, H, W)
        input_field = x[:, 0:1, :, :]  # (B*T, 1, H, W)

        # ── Encoder ────────────────────────────────────────────────
        skips = []
        h = x  # Use full input (field + mask)
        for down in self.encoder_blocks:
            skip, h = down(h)
            skips.append(skip)

        # ── Bottleneck ─────────────────────────────────────────────
        h = self.bottleneck(h)

        # ── Decoder ────────────────────────────────────────────────
        for i, up in enumerate(self.decoder_blocks):
            skip = skips[-(i + 1)]
            h = up(h, skip)

        # ── Output ─────────────────────────────────────────────────
        prediction = self.head(h)  # (B*T, 1, H, W)

        if self.use_tanh:
            prediction = torch.tanh(prediction)

        # Hard constraint: preserve observed pixels exactly
        if self.hard_merge:
            output = mask * input_field + (1.0 - mask) * prediction
        else:
            # Soft mode for GAN training — allow gradient flow everywhere
            output = prediction

        # Unfold time
        if has_time:
            output = output.reshape(B, T, self.out_channels, H, W)

        return output
