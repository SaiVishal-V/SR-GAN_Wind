"""
Prediction map visualization for WindGapGAN.

Generates side-by-side comparisons:
    - Ground Truth | Prediction | Mask
    - Best case / worst case examples
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_prediction_comparison(
    ground_truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray,
    save_path: Optional[str | Path] = None,
    title: str = "Gap Filling Result",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "viridis",
) -> plt.Figure:
    """
    Plot ground truth vs. prediction with mask overlay.

    Args:
        ground_truth: (H, W) ground truth field.
        prediction: (H, W) predicted field.
        mask: (H, W) observation mask (1=observed, 0=gap).
        save_path: Optional path to save figure.
        title: Figure title.
        vmin, vmax: Color scale limits.
        cmap: Colormap name.
    """
    if vmin is None:
        vmin = float(np.nanmin(ground_truth))
    if vmax is None:
        vmax = float(np.nanmax(ground_truth))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Ground Truth
    im0 = axes[0].imshow(ground_truth, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    axes[0].set_title("Ground Truth")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Masked Input (what the model sees)
    masked_input = ground_truth * mask
    masked_input_display = np.where(mask > 0, ground_truth, np.nan)
    im1 = axes[1].imshow(masked_input_display, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    axes[1].set_title("Masked Input (Observed)")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Prediction
    im2 = axes[2].imshow(prediction, cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
    axes[2].set_title("Prediction (Gap-Filled)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Mask
    im3 = axes[3].imshow(mask, cmap="gray", vmin=0, vmax=1, origin="lower")
    axes[3].set_title("Observation Mask")
    plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Prediction map saved to %s", save_path)
    return fig


def plot_best_worst_cases(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    save_dir: Optional[str | Path] = None,
    n_examples: int = 3,
) -> list[plt.Figure]:
    """
    Plot best and worst gap-filling examples based on RMSE.

    Args:
        predictions: (N, H, W) array of predictions.
        targets: (N, H, W) array of targets.
        masks: (N, H, W) array of masks.
        save_dir: Directory to save figures.
        n_examples: Number of best/worst examples to show.
    """
    N = predictions.shape[0]
    errors = np.zeros(N)

    for i in range(N):
        gap = (1 - masks[i]).astype(bool)
        if gap.any():
            errors[i] = np.sqrt(np.mean((predictions[i][gap] - targets[i][gap]) ** 2))
        else:
            errors[i] = 0.0

    # Sort by error
    sorted_idx = np.argsort(errors)
    best_idx = sorted_idx[:n_examples]
    worst_idx = sorted_idx[-n_examples:][::-1]

    figs = []
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    for label, indices in [("best", best_idx), ("worst", worst_idx)]:
        for rank, idx in enumerate(indices):
            save_path = (save_dir / f"{label}_case_{rank + 1}.png") if save_dir else None
            fig = plot_prediction_comparison(
                targets[idx],
                predictions[idx],
                masks[idx],
                save_path=save_path,
                title=f"{label.capitalize()} Case #{rank + 1} (RMSE={errors[idx]:.4f})",
            )
            figs.append(fig)

    return figs
