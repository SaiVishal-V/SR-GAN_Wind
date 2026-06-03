"""
Error map and histogram visualization for WindGapGAN.

Generates:
    - Spatial error maps (prediction - target)
    - Error histograms with distribution comparison
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_error_map(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    save_path: Optional[str | Path] = None,
    title: str = "Prediction Error Map",
) -> plt.Figure:
    """
    Plot spatial error map (prediction - target) on gap pixels.

    Args:
        prediction: (H, W) predicted field.
        target: (H, W) ground truth.
        mask: (H, W) observation mask.
    """
    error = prediction - target
    gap_error = np.where(mask < 1, error, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error map
    vmax = max(abs(np.nanmin(gap_error)), abs(np.nanmax(gap_error)))
    if np.isnan(vmax) or vmax < 1e-10:
        vmax = 1.0

    im0 = axes[0].imshow(gap_error, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
    axes[0].set_title("Error (Pred − Truth) [Gap Only]")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Absolute error
    abs_error = np.abs(gap_error)
    im1 = axes[1].imshow(abs_error, cmap="hot_r", origin="lower")
    axes[1].set_title("|Error| [Gap Only]")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Error map saved to %s", save_path)
    return fig


def plot_error_histogram(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    save_path: Optional[str | Path] = None,
    title: str = "Error Distribution",
    n_bins: int = 50,
) -> plt.Figure:
    """
    Plot error histogram and distribution comparison.

    Shows:
        1. Error distribution (pred - target) on gap pixels
        2. Value distribution comparison (prediction vs target)
    """
    gap_mask = (1 - mask).astype(bool)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error histogram
    if gap_mask.any():
        errors = prediction[gap_mask] - target[gap_mask]
        axes[0].hist(errors, bins=n_bins, color="#2196F3", alpha=0.7, edgecolor="black", linewidth=0.5)
        axes[0].axvline(0, color="red", linestyle="--", linewidth=1, label="Zero Error")
        axes[0].set_xlabel("Error (Pred − Truth)")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Error Distribution (Gap Pixels)")
        axes[0].legend()

        # Distribution comparison
        axes[1].hist(
            target[gap_mask], bins=n_bins, alpha=0.5, color="#4CAF50",
            label="Ground Truth", edgecolor="black", linewidth=0.5,
        )
        axes[1].hist(
            prediction[gap_mask], bins=n_bins, alpha=0.5, color="#FF9800",
            label="Prediction", edgecolor="black", linewidth=0.5,
        )
        axes[1].set_xlabel("Value")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Value Distribution Comparison (Gap Pixels)")
        axes[1].legend()
    else:
        axes[0].text(0.5, 0.5, "No gap pixels", ha="center", va="center", transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, "No gap pixels", ha="center", va="center", transform=axes[1].transAxes)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Error histogram saved to %s", save_path)
    return fig
