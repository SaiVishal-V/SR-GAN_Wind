"""
Loss functions for wind-speed super-resolution (V2).

All losses respect the ocean mask (hr_ocean_mask):
    - 1 = ocean → compute loss here
    - 0 = land → NEVER compute loss here

Scientific constraints:
    - No VGG perceptual loss (not RGB imagery)
    - No ImageNet normalization
    - Preserve wind-speed gradients and mesoscale structures

Fill value handling:
    - Never compute loss over fill values (-9999)
    - Never compute loss over land pixels

V2 additions:
    - Laplacian loss (sharpness)
    - Spectral loss (FFT frequency preservation)
    - Charbonnier loss (robust to outliers)
    - Multi-scale loss (multi-resolution consistency)
    - All new losses default to weight=0 (preserving V1 behavior)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked L1 loss — computed only over ocean pixels.

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W), 1=ocean, 0=land.

    Returns:
        Scalar loss value.
    """
    loss_map = torch.abs(pred - target)
    masked_loss = loss_map[mask == 1]

    if masked_loss.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return masked_loss.mean()


def masked_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked MSE loss — computed only over ocean pixels.

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W), 1=ocean, 0=land.

    Returns:
        Scalar loss value.
    """
    loss_map = (pred - target) ** 2
    masked_loss = loss_map[mask == 1]

    if masked_loss.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return masked_loss.mean()


def masked_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Masked RMSE — square root of masked MSE.

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W), 1=ocean, 0=land.

    Returns:
        Scalar RMSE value.
    """
    mse = masked_mse_loss(pred, target, mask)
    return torch.sqrt(mse + 1e-8)


def gradient_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Gradient loss — L1 between spatial gradients of prediction and target.
    Preserves wind-speed fronts and mesoscale structures.

    Computes: L_grad = L1(∇pred, ∇target) over ocean pixels.

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W), 1=ocean, 0=land.

    Returns:
        Scalar gradient loss value.
    """
    # Spatial gradients (finite differences)
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]

    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]

    # Mask gradients — both pixels in the difference must be ocean
    mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]
    mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    # Masked L1 on gradients
    loss_dy = torch.abs(pred_dy - target_dy)
    loss_dx = torch.abs(pred_dx - target_dx)

    masked_dy = loss_dy[mask_dy == 1]
    masked_dx = loss_dx[mask_dx == 1]

    total = 0.0
    count = 0

    if masked_dy.numel() > 0:
        total = total + masked_dy.sum()
        count += masked_dy.numel()
    if masked_dx.numel() > 0:
        total = total + masked_dx.sum()
        count += masked_dx.numel()

    if count == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return total / count


def laplacian_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Laplacian loss — L1 between Laplacian of prediction and target.
    Improves edge sharpness and second-order spatial consistency.

    Computes: L_lap = L1(∇²pred, ∇²target) over ocean pixels.

    The Laplacian kernel:
        [0,  1, 0]
        [1, -4, 1]
        [0,  1, 0]

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W), 1=ocean, 0=land.

    Returns:
        Scalar Laplacian loss value.
    """
    # Laplacian kernel
    kernel = torch.tensor(
        [[0, 1, 0],
         [1, -4, 1],
         [0, 1, 0]],
        dtype=pred.dtype, device=pred.device
    ).unsqueeze(0).unsqueeze(0)  # (1, 1, 3, 3)

    # Apply Laplacian filter
    pred_lap = F.conv2d(pred, kernel, padding=1)
    target_lap = F.conv2d(target, kernel, padding=1)

    # Erode mask by 1 pixel (Laplacian uses 3x3 neighborhood)
    mask_eroded = F.conv2d(
        mask,
        torch.ones(1, 1, 3, 3, dtype=mask.dtype, device=mask.device),
        padding=1,
    )
    mask_eroded = (mask_eroded >= 9).float()  # All 9 neighbors must be ocean

    # Masked L1
    loss_map = torch.abs(pred_lap - target_lap)
    masked_loss = loss_map[mask_eroded == 1]

    if masked_loss.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return masked_loss.mean()


def spectral_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Spectral loss — L1 between FFT magnitudes of prediction and target.
    Preserves spatial frequency distribution of wind fields.

    Computes: L_spec = L1(|FFT(pred * mask)|, |FFT(target * mask)|)

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W), 1=ocean, 0=land.

    Returns:
        Scalar spectral loss value.
    """
    # Zero out land pixels
    pred_masked = pred * mask
    target_masked = target * mask

    # 2D FFT
    pred_fft = torch.fft.fft2(pred_masked)
    target_fft = torch.fft.fft2(target_masked)

    # Magnitude spectrum (log scale for numerical stability)
    pred_mag = torch.log(torch.abs(pred_fft) + 1e-8)
    target_mag = torch.log(torch.abs(target_fft) + 1e-8)

    # L1 in log-magnitude space
    loss = torch.abs(pred_mag - target_mag).mean()

    return loss


def charbonnier_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Charbonnier loss — smooth approximation of L1, more robust to outliers.

    Computes: L_char = sqrt((pred - target)^2 + eps^2) over ocean pixels.

    Commonly used in modern super-resolution (e.g., SwinIR, EDSR).

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W), 1=ocean, 0=land.
        eps: Epsilon for smooth differentiability.

    Returns:
        Scalar Charbonnier loss value.
    """
    diff_sq = (pred - target) ** 2
    loss_map = torch.sqrt(diff_sq + eps ** 2)
    masked_loss = loss_map[mask == 1]

    if masked_loss.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)

    return masked_loss.mean()


def multiscale_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Multi-scale loss — masked L1 at 1×, 0.5×, and 0.25× resolutions.

    Encourages consistency across spatial scales.

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W), 1=ocean, 0=land.

    Returns:
        Scalar multi-scale loss (averaged across scales).
    """
    scales = [1.0, 0.5, 0.25]
    total_loss = torch.tensor(0.0, device=pred.device, requires_grad=True)

    for scale in scales:
        if scale < 1.0:
            p = F.interpolate(pred, scale_factor=scale, mode="bilinear", align_corners=False)
            t = F.interpolate(target, scale_factor=scale, mode="bilinear", align_corners=False)
            m = F.interpolate(mask, scale_factor=scale, mode="nearest")
        else:
            p, t, m = pred, target, mask

        total_loss = total_loss + masked_l1_loss(p, t, m)

    return total_loss / len(scales)


def adversarial_loss_g(
    fake_preds: torch.Tensor,
) -> torch.Tensor:
    """
    Generator adversarial loss — BCE with targets=1 (fool discriminator).

    Args:
        fake_preds: Discriminator output on SR images.

    Returns:
        Scalar generator adversarial loss.
    """
    target = torch.ones_like(fake_preds)
    return F.binary_cross_entropy_with_logits(fake_preds, target)


def adversarial_loss_d(
    real_preds: torch.Tensor,
    fake_preds: torch.Tensor,
) -> torch.Tensor:
    """
    Discriminator loss — BCE with real=1, fake=0.

    Args:
        real_preds: Discriminator output on real HR images.
        fake_preds: Discriminator output on SR (fake) images.

    Returns:
        Scalar discriminator loss.
    """
    real_loss = F.binary_cross_entropy_with_logits(
        real_preds, torch.ones_like(real_preds)
    )
    fake_loss = F.binary_cross_entropy_with_logits(
        fake_preds, torch.zeros_like(fake_preds)
    )
    return (real_loss + fake_loss) / 2


class GeneratorLoss(nn.Module):
    """
    Combined generator loss (V2):
        L_G = λ1 * pixel + λ2 * adversarial + λ3 * gradient
            + λ4 * laplacian + λ5 * spectral + λ6 * multiscale

    Pixel loss can be L1 or Charbonnier (configurable).

    All new losses default to weight=0.0 to preserve V1 behavior.

    Args:
        pixel_weight: Weight for pixel loss (λ1).
        adversarial_weight: Weight for adversarial loss (λ2).
        gradient_weight: Weight for gradient loss (λ3).
        laplacian_weight: Weight for Laplacian loss (λ4).
        spectral_weight: Weight for spectral loss (λ5).
        multiscale_weight: Weight for multi-scale loss (λ6).
        pixel_loss_type: 'l1' or 'charbonnier'.
    """

    def __init__(
        self,
        pixel_weight: float = 1.0,
        adversarial_weight: float = 1e-3,
        gradient_weight: float = 0.1,
        laplacian_weight: float = 0.0,
        spectral_weight: float = 0.0,
        multiscale_weight: float = 0.0,
        pixel_loss_type: str = "l1",
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.adversarial_weight = adversarial_weight
        self.gradient_weight = gradient_weight
        self.laplacian_weight = laplacian_weight
        self.spectral_weight = spectral_weight
        self.multiscale_weight = multiscale_weight
        self.pixel_loss_type = pixel_loss_type

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        fake_preds: torch.Tensor = None,
    ) -> dict:
        """
        Compute combined generator loss.

        Args:
            pred: SR output (B, 1, H, W).
            target: HR ground truth (B, 1, H, W).
            mask: Ocean mask (B, 1, H, W).
            fake_preds: Discriminator output on SR images (None during pretraining).

        Returns:
            Dictionary with total loss and individual components.
        """
        # Pixel loss
        if self.pixel_loss_type == "charbonnier":
            pixel = charbonnier_loss(pred, target, mask)
        else:
            pixel = masked_l1_loss(pred, target, mask)

        grad = gradient_loss(pred, target, mask)

        total = self.pixel_weight * pixel + self.gradient_weight * grad

        losses = {
            "pixel_loss": pixel,
            "gradient_loss": grad,
        }

        # Laplacian loss
        if self.laplacian_weight > 0:
            lap = laplacian_loss(pred, target, mask)
            total = total + self.laplacian_weight * lap
            losses["laplacian_loss"] = lap

        # Spectral loss
        if self.spectral_weight > 0:
            spec = spectral_loss(pred, target, mask)
            total = total + self.spectral_weight * spec
            losses["spectral_loss"] = spec

        # Multi-scale loss
        if self.multiscale_weight > 0:
            ms = multiscale_loss(pred, target, mask)
            total = total + self.multiscale_weight * ms
            losses["multiscale_loss"] = ms

        # Adversarial loss
        if fake_preds is not None:
            adv = adversarial_loss_g(fake_preds)
            total = total + self.adversarial_weight * adv
            losses["adversarial_loss"] = adv

        losses["total_loss"] = total
        return losses
