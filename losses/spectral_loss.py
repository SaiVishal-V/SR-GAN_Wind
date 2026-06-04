"""
Spectral loss for WindGapGAN.

FFT-based loss that penalizes differences in the spatial frequency
content between prediction and target.  This is critical for
preventing the blurry, low-frequency outputs that L1 loss alone
produces.

The model is forced to generate high-frequency details (sharp edges,
fine wind patterns) that match the target's power spectrum.

Reference:
    Fuoli et al., "Fourier Space Losses for Efficient Perceptual Image
    Super-Resolution" (ICCV 2021)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SpectralLoss(nn.Module):
    """
    FFT-based spectral matching loss.

    loss = || log(|FFT(pred)|) - log(|FFT(target)|) ||₁

    Operates on 2D spatial FFT of each sample independently.
    Uses log-magnitude for numerical stability and to weight
    all frequency bands more equally.

    Args:
        log_scale: If True, compare log-magnitudes (recommended).
        reduction: 'mean' or 'sum'.
    """

    def __init__(
        self,
        log_scale: bool = True,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.log_scale = log_scale
        self.reduction = reduction

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute spectral loss.

        Args:
            prediction: (B, C, H, W) predicted field.
            target: (B, C, H, W) ground truth field.
            mask: Optional (B, C, H, W) mask. If provided, apply before FFT
                to focus on gap regions. Not used for masking FFT output
                (that would break frequency analysis).

        Returns:
            Scalar loss.
        """
        # Handle 5D temporal input
        if prediction.ndim == 5:
            B, T, C, H, W = prediction.shape
            prediction = prediction.reshape(B * T, C, H, W)
            target = target.reshape(B * T, C, H, W)

        # 2D FFT
        pred_fft = torch.fft.rfft2(prediction, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")

        # Magnitude spectrum
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        if self.log_scale:
            eps = 1e-7
            pred_mag = torch.log(pred_mag + eps)
            target_mag = torch.log(target_mag + eps)

        # L1 loss on magnitude spectrum
        if self.reduction == "mean":
            loss = torch.mean(torch.abs(pred_mag - target_mag))
        else:
            loss = torch.sum(torch.abs(pred_mag - target_mag))

        return loss
