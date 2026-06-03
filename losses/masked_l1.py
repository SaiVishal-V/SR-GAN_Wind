"""
Masked L1 loss for gap filling.

Computes L1 loss ONLY on gap (missing) pixels, with optional
regularization on observed pixels to ensure they remain unchanged.

This is the primary loss function for all phases.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskedL1Loss(nn.Module):
    """
    L1 loss computed on gap pixels only.

    loss = || (1 - mask) ⊙ (prediction - target) ||₁ / N_gap
         + α * || mask ⊙ (prediction - target) ||₁ / N_obs

    Where:
        mask = 1 for observed, 0 for missing
        N_gap = number of gap pixels
        N_obs = number of observed pixels
        α = observed_weight (small, e.g. 0.01, to penalize observed distortion)

    The observed term should be near-zero if the model uses masked merge,
    but serves as a safety regularizer.
    """

    def __init__(self, observed_weight: float = 0.01) -> None:
        """
        Args:
            observed_weight: Weight for the observed-pixel preservation term.
                Set to 0 to disable.
        """
        super().__init__()
        self.observed_weight = observed_weight

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            prediction: Model output, shape (...).
            target: Ground truth, shape (...).
            mask: Observation mask, shape (...). 1=observed, 0=gap.

        Returns:
            Scalar loss value.
        """
        gap_mask = 1.0 - mask  # 1 where gap, 0 where observed
        error = torch.abs(prediction - target)

        # Gap loss
        n_gap = gap_mask.sum().clamp(min=1.0)
        gap_loss = (gap_mask * error).sum() / n_gap

        # Observed preservation loss (optional)
        if self.observed_weight > 0:
            n_obs = mask.sum().clamp(min=1.0)
            obs_loss = (mask * error).sum() / n_obs
            total = gap_loss + self.observed_weight * obs_loss
        else:
            total = gap_loss

        return total
