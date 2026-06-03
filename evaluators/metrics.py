"""
Gap-filling evaluation metrics for WindGapGAN.

Two metric groups:
    Group 1 — Gap Reconstruction: computed on missing pixels only (primary metrics)
    Group 2 — Observation Preservation: computed on observed pixels only

Additional:
    - Distribution metrics: KL Divergence, Wasserstein Distance
    - Error stratification by value regime (e.g., wind-speed bins)

Never evaluates over all pixels indiscriminately.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class GapFillingMetrics:
    """
    Compute gap-filling evaluation metrics.

    All metrics require prediction, ground truth, and observation mask.
    """

    @staticmethod
    def rmse(prediction: np.ndarray, target: np.ndarray, region_mask: np.ndarray) -> float:
        """Root Mean Square Error on specified region."""
        valid = region_mask.astype(bool)
        if not valid.any():
            return float("nan")
        diff = prediction[valid] - target[valid]
        return float(np.sqrt(np.mean(diff ** 2)))

    @staticmethod
    def mae(prediction: np.ndarray, target: np.ndarray, region_mask: np.ndarray) -> float:
        """Mean Absolute Error on specified region."""
        valid = region_mask.astype(bool)
        if not valid.any():
            return float("nan")
        return float(np.mean(np.abs(prediction[valid] - target[valid])))

    @staticmethod
    def bias(prediction: np.ndarray, target: np.ndarray, region_mask: np.ndarray) -> float:
        """Mean Bias (prediction - target) on specified region."""
        valid = region_mask.astype(bool)
        if not valid.any():
            return float("nan")
        return float(np.mean(prediction[valid] - target[valid]))

    @staticmethod
    def correlation(prediction: np.ndarray, target: np.ndarray, region_mask: np.ndarray) -> float:
        """Pearson correlation coefficient on specified region."""
        valid = region_mask.astype(bool)
        if not valid.any() or valid.sum() < 2:
            return float("nan")
        pred_vals = prediction[valid]
        target_vals = target[valid]
        if np.std(pred_vals) < 1e-10 or np.std(target_vals) < 1e-10:
            return float("nan")
        corr = float(np.corrcoef(pred_vals, target_vals)[0, 1])
        return corr

    @staticmethod
    def kl_divergence(
        prediction: np.ndarray,
        target: np.ndarray,
        region_mask: np.ndarray,
        n_bins: int = 50,
    ) -> float:
        """
        KL Divergence between prediction and target distributions.

        Uses histogram binning to estimate distributions.
        """
        valid = region_mask.astype(bool)
        if not valid.any() or valid.sum() < 10:
            return float("nan")

        pred_vals = prediction[valid]
        target_vals = target[valid]

        # Shared bin edges
        all_vals = np.concatenate([pred_vals, target_vals])
        bin_edges = np.linspace(all_vals.min(), all_vals.max(), n_bins + 1)

        pred_hist, _ = np.histogram(pred_vals, bins=bin_edges, density=True)
        target_hist, _ = np.histogram(target_vals, bins=bin_edges, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-10
        pred_hist = pred_hist + eps
        target_hist = target_hist + eps

        # Normalize to valid probability distributions
        pred_hist = pred_hist / pred_hist.sum()
        target_hist = target_hist / target_hist.sum()

        kl = float(np.sum(target_hist * np.log(target_hist / pred_hist)))
        return kl

    @staticmethod
    def wasserstein_distance(
        prediction: np.ndarray,
        target: np.ndarray,
        region_mask: np.ndarray,
    ) -> float:
        """Wasserstein (Earth Mover's) distance between distributions."""
        valid = region_mask.astype(bool)
        if not valid.any() or valid.sum() < 10:
            return float("nan")
        return float(stats.wasserstein_distance(prediction[valid], target[valid]))


def compute_gap_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """
    Compute Group 1 metrics (gap reconstruction) — primary metrics.

    Args:
        prediction: Predicted field, shape (T, H, W) or (H, W).
        target: Ground truth field, same shape.
        mask: Observation mask (1=observed, 0=gap), same shape.

    Returns:
        Dict of gap metrics.
    """
    gap_mask = (1.0 - mask).astype(bool).astype(np.float32)

    return {
        "rmse_gap": GapFillingMetrics.rmse(prediction, target, gap_mask),
        "mae_gap": GapFillingMetrics.mae(prediction, target, gap_mask),
        "bias_gap": GapFillingMetrics.bias(prediction, target, gap_mask),
        "corr_gap": GapFillingMetrics.correlation(prediction, target, gap_mask),
        "kl_div_gap": GapFillingMetrics.kl_divergence(prediction, target, gap_mask),
        "wasserstein_gap": GapFillingMetrics.wasserstein_distance(prediction, target, gap_mask),
    }


def compute_observed_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    """
    Compute Group 2 metrics (observation preservation).

    These should be near-zero if the model preserves observed pixels correctly.

    Args:
        prediction: Predicted field.
        target: Ground truth field.
        mask: Observation mask (1=observed, 0=gap).

    Returns:
        Dict of observation preservation metrics.
    """
    obs_mask = mask.astype(np.float32)

    return {
        "rmse_observed": GapFillingMetrics.rmse(prediction, target, obs_mask),
        "mae_observed": GapFillingMetrics.mae(prediction, target, obs_mask),
        "bias_observed": GapFillingMetrics.bias(prediction, target, obs_mask),
        "corr_observed": GapFillingMetrics.correlation(prediction, target, obs_mask),
    }


def compute_stratified_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    regimes: list[list[Optional[float]]] | None = None,
) -> dict[str, dict[str, float]]:
    """
    Compute error-stratified metrics by value regime.

    Args:
        prediction: Predicted field.
        target: Ground truth field.
        mask: Observation mask.
        regimes: List of [low, high] value ranges. None in high means infinity.
            Default: [[0,5], [5,10], [10,15], [15,None]] (wind speed m/s).

    Returns:
        Nested dict: {regime_label: {metric_name: value}}.
    """
    if regimes is None:
        regimes = [[0, 5], [5, 10], [10, 15], [15, None]]

    gap_mask = (1.0 - mask).astype(bool)
    results = {}

    for regime in regimes:
        low, high = regime
        label = f"{low}-{high if high is not None else 'inf'}"

        # Find gap pixels in this regime (based on ground truth)
        in_range = np.ones_like(target, dtype=bool)
        if low is not None:
            in_range &= target >= low
        if high is not None:
            in_range &= target < high

        regime_mask = (gap_mask & in_range).astype(np.float32)
        n_pixels = int(regime_mask.sum())

        if n_pixels < 2:
            results[label] = {
                "rmse": float("nan"),
                "mae": float("nan"),
                "bias": float("nan"),
                "corr": float("nan"),
                "n_pixels": n_pixels,
            }
        else:
            results[label] = {
                "rmse": GapFillingMetrics.rmse(prediction, target, regime_mask),
                "mae": GapFillingMetrics.mae(prediction, target, regime_mask),
                "bias": GapFillingMetrics.bias(prediction, target, regime_mask),
                "corr": GapFillingMetrics.correlation(prediction, target, regime_mask),
                "n_pixels": n_pixels,
            }

    return results


def compute_all_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    regimes: list[list[Optional[float]]] | None = None,
) -> dict[str, Any]:
    """
    Compute all metrics: gap, observed, and stratified.

    Args:
        prediction: Predicted field.
        target: Ground truth field.
        mask: Observation mask.
        regimes: Value regime bins for stratification.

    Returns:
        Comprehensive metrics dictionary.
    """
    return {
        "gap": compute_gap_metrics(prediction, target, mask),
        "observed": compute_observed_metrics(prediction, target, mask),
        "stratified": compute_stratified_metrics(prediction, target, mask, regimes),
    }
