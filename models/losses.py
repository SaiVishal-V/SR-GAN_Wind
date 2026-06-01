"""
Loss functions for wind-speed super-resolution.

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
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    Combined generator loss:
        L_G = λ1 * masked_L1 + λ2 * adversarial + λ3 * gradient_loss

    Args:
        pixel_weight: Weight for masked L1 loss (λ1).
        adversarial_weight: Weight for adversarial loss (λ2).
        gradient_weight: Weight for gradient loss (λ3).
    """

    def __init__(
        self,
        pixel_weight: float = 1.0,
        adversarial_weight: float = 1e-3,
        gradient_weight: float = 0.1,
    ):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.adversarial_weight = adversarial_weight
        self.gradient_weight = gradient_weight

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
        l1 = masked_l1_loss(pred, target, mask)
        grad = gradient_loss(pred, target, mask)

        total = self.pixel_weight * l1 + self.gradient_weight * grad

        losses = {
            "pixel_loss": l1,
            "gradient_loss": grad,
        }

        if fake_preds is not None:
            adv = adversarial_loss_g(fake_preds)
            total = total + self.adversarial_weight * adv
            losses["adversarial_loss"] = adv

        losses["total_loss"] = total
        return losses
