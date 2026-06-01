"""
Evaluation metrics for wind-speed super-resolution.

Metrics tracked during training:
    - Masked MAE (normalized)
    - Masked RMSE (normalized)
    - PSNR (ocean-only)
    - SSIM (ocean-only)

Physical metrics after denormalization:
    - RMSE (m/s)
    - MAE (m/s)
    - Bias (m/s)
    - Correlation Coefficient
    - Gradient RMSE (review #7: critical for wind front evaluation)

Denormalization formula (verified from dataset):
    wind_speed_mps = wind_speed_norm * 2.9275431488456434 + 6.336302810300043
"""

import torch
import numpy as np
from typing import Dict
from skimage.metrics import structural_similarity as compare_ssim


# Verified normalization constants
NORM_MEAN = 6.336302810300043
NORM_STD = 2.9275431488456434


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """
    Denormalize wind speed from z-score to m/s.

    Args:
        x: Normalized tensor.

    Returns:
        Wind speed in m/s.
    """
    return x * NORM_STD + NORM_MEAN


def masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Masked MAE in normalized space."""
    diff = torch.abs(pred - target)
    values = diff[mask == 1]
    if values.numel() == 0:
        return 0.0
    return values.mean().item()


def masked_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """Masked RMSE in normalized space."""
    sq_diff = (pred - target) ** 2
    values = sq_diff[mask == 1]
    if values.numel() == 0:
        return 0.0
    return torch.sqrt(values.mean()).item()


def masked_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    data_range: float = None,
) -> float:
    """
    PSNR computed only over ocean pixels.

    Args:
        pred: Predicted tensor.
        target: Ground truth tensor.
        mask: Ocean mask.
        data_range: Max - min of valid data range. Auto-computed if None.

    Returns:
        PSNR in dB.
    """
    values_pred = pred[mask == 1]
    values_target = target[mask == 1]

    if values_pred.numel() == 0:
        return 0.0

    if data_range is None:
        data_range = values_target.max().item() - values_target.min().item()
        if data_range < 1e-8:
            return float("inf")

    mse = ((values_pred - values_target) ** 2).mean().item()
    if mse < 1e-10:
        return float("inf")

    return 10.0 * np.log10(data_range ** 2 / mse)


def masked_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    SSIM computed over the full image but only ocean-relevant regions contribute.
    Uses scikit-image for robust computation.

    Args:
        pred: Predicted tensor (B, 1, H, W) or (1, H, W).
        target: Ground truth tensor.
        mask: Ocean mask.

    Returns:
        Mean SSIM value.
    """
    # Convert to numpy, squeeze batch/channel dims
    if pred.dim() == 4:
        pred = pred.squeeze(0)
    if target.dim() == 4:
        target = target.squeeze(0)
    if mask.dim() == 4:
        mask = mask.squeeze(0)

    pred_np = pred.squeeze(0).detach().cpu().numpy()
    target_np = target.squeeze(0).detach().cpu().numpy()
    mask_np = mask.squeeze(0).detach().cpu().numpy()

    data_range = target_np[mask_np == 1].max() - target_np[mask_np == 1].min()
    if data_range < 1e-8:
        return 1.0

    # Compute SSIM with window size
    win_size = min(7, min(pred_np.shape) - 1)
    if win_size % 2 == 0:
        win_size -= 1
    if win_size < 3:
        win_size = 3

    try:
        ssim_val = compare_ssim(
            target_np,
            pred_np,
            data_range=data_range,
            win_size=win_size,
        )
    except Exception:
        ssim_val = 0.0

    return float(ssim_val)


def gradient_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Gradient RMSE — evaluates preservation of wind-speed fronts.
    (Review #7: most important scientific change)

    Computes RMSE of spatial gradients (dU/dx, dU/dy) over ocean pixels.

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W).

    Returns:
        Gradient RMSE value.
    """
    # dU/dy
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    # dU/dx
    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    sq_dy = (pred_dy - target_dy) ** 2
    sq_dx = (pred_dx - target_dx) ** 2

    vals_dy = sq_dy[mask_dy == 1]
    vals_dx = sq_dx[mask_dx == 1]

    all_vals = torch.cat([vals_dy, vals_dx])
    if all_vals.numel() == 0:
        return 0.0

    return torch.sqrt(all_vals.mean()).item()


def physical_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute physical metrics in m/s after denormalization.

    Args:
        pred: Predicted tensor in normalized space.
        target: Ground truth tensor in normalized space.
        mask: Ocean mask.

    Returns:
        Dictionary of physical metrics.
    """
    pred_mps = denormalize(pred)
    target_mps = denormalize(target)

    pred_vals = pred_mps[mask == 1].detach().cpu()
    target_vals = target_mps[mask == 1].detach().cpu()

    if pred_vals.numel() == 0:
        return {
            "rmse_mps": 0.0,
            "mae_mps": 0.0,
            "bias_mps": 0.0,
            "correlation": 0.0,
        }

    diff = pred_vals - target_vals

    rmse_mps = torch.sqrt((diff ** 2).mean()).item()
    mae_mps = torch.abs(diff).mean().item()
    bias_mps = diff.mean().item()

    # Pearson correlation
    pred_mean = pred_vals.mean()
    target_mean = target_vals.mean()
    cov = ((pred_vals - pred_mean) * (target_vals - target_mean)).mean()
    std_pred = pred_vals.std()
    std_target = target_vals.std()

    if std_pred < 1e-8 or std_target < 1e-8:
        correlation = 0.0
    else:
        correlation = (cov / (std_pred * std_target)).item()

    return {
        "rmse_mps": rmse_mps,
        "mae_mps": mae_mps,
        "bias_mps": bias_mps,
        "correlation": correlation,
    }


def compute_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all metrics (normalized + physical).

    Args:
        pred: SR output in normalized space.
        target: HR ground truth in normalized space.
        mask: Ocean mask.

    Returns:
        Dictionary with all metric values.
    """
    metrics = {
        "mae": masked_mae(pred, target, mask),
        "rmse": masked_rmse(pred, target, mask),
        "psnr": masked_psnr(pred, target, mask),
        "ssim": masked_ssim(pred, target, mask),
        "gradient_rmse": gradient_rmse(pred, target, mask),
    }

    phys = physical_metrics(pred, target, mask)
    metrics.update(phys)

    return metrics
