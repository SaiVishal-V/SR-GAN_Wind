"""
Self-Attention block for U-Net bottleneck.

Provides long-range spatial dependencies that convolutions alone
cannot capture, essential for filling large contiguous gaps.

Reference:
    Zhang et al., "Self-Attention Generative Adversarial Networks" (ICML 2019)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention2d(nn.Module):
    """
    Self-attention mechanism for 2D feature maps.

    Computes attention over all spatial positions, allowing each
    position to attend to every other position. Uses learned
    query, key, value projections with a learnable scaling factor.

    Args:
        channels: Number of input/output channels.
        reduction: Channel reduction factor for Q, K projections
            (saves memory with minimal quality loss).
    """

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()

        self.channels = channels
        mid_channels = max(channels // reduction, 1)

        self.query = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.key = nn.Conv2d(channels, mid_channels, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)

        # Learnable scaling factor (starts at 0 → residual pass-through)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) feature map.

        Returns:
            (B, C, H, W) — attention-weighted features + residual.
        """
        B, C, H, W = x.shape
        N = H * W  # number of spatial positions

        # Project to query, key, value
        q = self.query(x).view(B, -1, N)             # (B, C', N)
        k = self.key(x).view(B, -1, N)               # (B, C', N)
        v = self.value(x).view(B, -1, N)              # (B, C, N)

        # Attention weights: softmax(Q^T K / sqrt(d))
        attn = torch.bmm(q.permute(0, 2, 1), k)      # (B, N, N)
        attn = F.softmax(attn / (q.shape[1] ** 0.5), dim=-1)

        # Weighted sum of values
        out = torch.bmm(v, attn.permute(0, 2, 1))    # (B, C, N)
        out = out.view(B, C, H, W)

        # Residual connection with learnable scaling
        return self.gamma * out + x
