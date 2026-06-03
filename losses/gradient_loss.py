"""
Spatial gradient loss for gap filling.

Penalizes discontinuities at gap boundaries by comparing spatial
gradients of the prediction and target. Uses Sobel-like filters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientLoss(nn.Module):
    """
    Spatial gradient consistency loss.

    Computes L1 difference between spatial gradients (dx, dy) of
    prediction and target. This encourages smooth transitions
    at gap boundaries.

    loss = || ∇x(pred) - ∇x(target) ||₁ + || ∇y(pred) - ∇y(target) ||₁
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            prediction: (B, C, H, W) or (B, T, C, H, W).
            target: Same shape as prediction.
            mask: Optional mask. If provided, compute loss only where mask=0 (gap).

        Returns:
            Scalar loss.
        """
        # Handle temporal dimension
        if prediction.ndim == 5:
            B, T, C, H, W = prediction.shape
            prediction = prediction.reshape(B * T, C, H, W)
            target = target.reshape(B * T, C, H, W)
            if mask is not None:
                mask = mask.reshape(B * T, C, H, W)

        # Compute gradients via finite differences
        pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
        pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_dx = torch.abs(pred_dx - target_dx).mean()
        loss_dy = torch.abs(pred_dy - target_dy).mean()

        return loss_dx + loss_dy
