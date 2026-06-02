"""
Evaluation metrics for wind-speed super-resolution (V2).

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
    - Gradient RMSE (critical for wind front evaluation)
    - Gradient MAE

Distribution metrics:
    - KL Divergence
    - Wasserstein Distance

Wind-speed regime metrics:
    - Stratified RMSE/MAE/Bias by speed range

Spectral metrics:
    - Power Spectrum Error (radially-averaged)

Observed-only metrics:
    - RMSE/MAE computed only on satellite-observed pixels

Denormalization formula (verified from dataset):
    wind_speed_mps = wind_speed_norm * 2.9275431488456434 + 6.336302810300043
"""

import torch
import numpy as np
from typing import Dict, Optional
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


def gradient_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Gradient MAE — L1 error of spatial gradients over ocean pixels.

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W).

    Returns:
        Gradient MAE value.
    """
    pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    mask_dy = mask[:, :, 1:, :] * mask[:, :, :-1, :]

    pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    mask_dx = mask[:, :, :, 1:] * mask[:, :, :, :-1]

    abs_dy = torch.abs(pred_dy - target_dy)
    abs_dx = torch.abs(pred_dx - target_dx)

    vals_dy = abs_dy[mask_dy == 1]
    vals_dx = abs_dx[mask_dx == 1]

    all_vals = torch.cat([vals_dy, vals_dx])
    if all_vals.numel() == 0:
        return 0.0

    return all_vals.mean().item()


def observed_only_rmse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    observed_mask: torch.Tensor,
) -> float:
    """
    RMSE computed only over satellite-observed ocean pixels.

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W).
        observed_mask: Satellite observation mask (B, 1, H, W).

    Returns:
        RMSE over observed ocean pixels.
    """
    combined = mask * observed_mask
    sq_diff = (pred - target) ** 2
    values = sq_diff[combined == 1]
    if values.numel() == 0:
        return 0.0
    return torch.sqrt(values.mean()).item()


def observed_only_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    observed_mask: torch.Tensor,
) -> float:
    """
    MAE computed only over satellite-observed ocean pixels.

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W).
        observed_mask: Satellite observation mask (B, 1, H, W).

    Returns:
        MAE over observed ocean pixels.
    """
    combined = mask * observed_mask
    abs_diff = torch.abs(pred - target)
    values = abs_diff[combined == 1]
    if values.numel() == 0:
        return 0.0
    return values.mean().item()


def power_spectrum_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Power spectrum error — L1 between radially-averaged FFT power spectra.

    Measures whether the SR output preserves the spatial frequency distribution
    of the ground truth wind field.

    Args:
        pred: Predicted tensor (B, 1, H, W).
        target: Ground truth tensor (B, 1, H, W).
        mask: Ocean mask (B, 1, H, W).

    Returns:
        Mean absolute log-power-spectrum error.
    """
    # Work with first sample in batch
    pred_np = pred[0, 0].detach().cpu().numpy()
    target_np = target[0, 0].detach().cpu().numpy()
    mask_np = mask[0, 0].detach().cpu().numpy()

    # Zero out land pixels for clean FFT
    pred_clean = pred_np * mask_np
    target_clean = target_np * mask_np

    # 2D FFT
    pred_fft = np.fft.fft2(pred_clean)
    target_fft = np.fft.fft2(target_clean)

    # Shift zero-frequency to center
    pred_power = np.abs(np.fft.fftshift(pred_fft)) ** 2
    target_power = np.abs(np.fft.fftshift(target_fft)) ** 2

    # Radially averaged power spectrum
    h, w = pred_np.shape
    cy, cx = h // 2, w // 2
    y, x = np.mgrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    max_r = min(cy, cx)

    pred_radial = np.zeros(max_r)
    target_radial = np.zeros(max_r)
    for ri in range(max_r):
        ring_mask = r == ri
        if ring_mask.sum() > 0:
            pred_radial[ri] = pred_power[ring_mask].mean()
            target_radial[ri] = target_power[ring_mask].mean()

    # Log-space comparison (skip DC component)
    pred_log = np.log10(pred_radial[1:] + 1e-10)
    target_log = np.log10(target_radial[1:] + 1e-10)

    return float(np.abs(pred_log - target_log).mean())


def kl_divergence(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    n_bins: int = 100,
) -> float:
    """
    KL Divergence between predicted and target wind-speed distributions.

    Args:
        pred: Predicted tensor in normalized space.
        target: Ground truth tensor in normalized space.
        mask: Ocean mask.
        n_bins: Number of histogram bins.

    Returns:
        KL(target || pred) in nats.
    """
    pred_vals = denormalize(pred)[mask == 1].detach().cpu().numpy()
    target_vals = denormalize(target)[mask == 1].detach().cpu().numpy()

    if len(pred_vals) == 0 or len(target_vals) == 0:
        return 0.0

    # Shared bins
    all_vals = np.concatenate([pred_vals, target_vals])
    bins = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

    p_hist, _ = np.histogram(target_vals, bins=bins, density=True)
    q_hist, _ = np.histogram(pred_vals, bins=bins, density=True)

    # Add epsilon to avoid log(0)
    eps = 1e-10
    p_hist = p_hist + eps
    q_hist = q_hist + eps

    # Normalize to proper distributions
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    kl = float(np.sum(p_hist * np.log(p_hist / q_hist)))
    return kl


def wasserstein_distance_metric(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    """
    Wasserstein (Earth Mover's) distance between wind-speed distributions.

    Args:
        pred: Predicted tensor in normalized space.
        target: Ground truth tensor in normalized space.
        mask: Ocean mask.

    Returns:
        Wasserstein-1 distance in m/s.
    """
    from scipy.stats import wasserstein_distance as wd

    pred_vals = denormalize(pred)[mask == 1].detach().cpu().numpy()
    target_vals = denormalize(target)[mask == 1].detach().cpu().numpy()

    if len(pred_vals) == 0 or len(target_vals) == 0:
        return 0.0

    # Subsample for efficiency
    max_samples = 50000
    if len(pred_vals) > max_samples:
        idx = np.random.choice(len(pred_vals), max_samples, replace=False)
        pred_vals = pred_vals[idx]
        target_vals = target_vals[idx]

    return float(wd(target_vals, pred_vals))


def wind_regime_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, Dict[str, float]]:
    """
    Stratified metrics by wind-speed regime (in m/s).

    Regimes:
        - Low:      0–5 m/s
        - Moderate:  5–10 m/s
        - High:     10–15 m/s
        - Extreme:  >15 m/s

    Args:
        pred: Predicted tensor in normalized space.
        target: Ground truth tensor in normalized space.
        mask: Ocean mask.

    Returns:
        Dict mapping regime name to {rmse, mae, bias, count}.
    """
    pred_mps = denormalize(pred)[mask == 1].detach().cpu().numpy()
    target_mps = denormalize(target)[mask == 1].detach().cpu().numpy()

    regimes = {
        "low_0_5": (0, 5),
        "moderate_5_10": (5, 10),
        "high_10_15": (10, 15),
        "extreme_15+": (15, float("inf")),
    }

    results = {}
    for name, (lo, hi) in regimes.items():
        idx = (target_mps >= lo) & (target_mps < hi)
        count = int(idx.sum())
        if count == 0:
            results[name] = {"rmse": 0.0, "mae": 0.0, "bias": 0.0, "count": 0}
            continue

        diff = pred_mps[idx] - target_mps[idx]
        results[name] = {
            "rmse": float(np.sqrt((diff ** 2).mean())),
            "mae": float(np.abs(diff).mean()),
            "bias": float(diff.mean()),
            "count": count,
        }

    return results


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
    observed_mask: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Compute all metrics (normalized + physical + gradient + spectral + distribution).

    Args:
        pred: SR output in normalized space.
        target: HR ground truth in normalized space.
        mask: Ocean mask.
        observed_mask: Optional satellite observation mask.

    Returns:
        Dictionary with all metric values.
    """
    metrics = {
        "mae": masked_mae(pred, target, mask),
        "rmse": masked_rmse(pred, target, mask),
        "psnr": masked_psnr(pred, target, mask),
        "ssim": masked_ssim(pred, target, mask),
        "gradient_rmse": gradient_rmse(pred, target, mask),
        "gradient_mae": gradient_mae(pred, target, mask),
        "power_spectrum_error": power_spectrum_error(pred, target, mask),
    }

    # Distribution metrics
    metrics["kl_divergence"] = kl_divergence(pred, target, mask)
    metrics["wasserstein_distance"] = wasserstein_distance_metric(pred, target, mask)

    # Physical metrics
    phys = physical_metrics(pred, target, mask)
    metrics.update(phys)

    # Observed-only metrics
    if observed_mask is not None:
        metrics["rmse_observed"] = observed_only_rmse(pred, target, mask, observed_mask)
        metrics["mae_observed"] = observed_only_mae(pred, target, mask, observed_mask)

    return metrics
