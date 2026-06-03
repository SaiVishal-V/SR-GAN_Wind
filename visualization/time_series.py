"""
Time-series visualization for WindGapGAN.

Plots temporal evolution of predictions vs ground truth
at specific spatial locations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_time_series(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    pixel_locations: list[tuple[int, int]] | None = None,
    time_labels: list | None = None,
    save_path: Optional[str | Path] = None,
    title: str = "Temporal Evolution",
) -> plt.Figure:
    """
    Plot prediction vs ground truth time series at specific pixels.

    Args:
        predictions: (T, H, W) predicted field.
        targets: (T, H, W) ground truth.
        masks: (T, H, W) observation mask.
        pixel_locations: List of (row, col) tuples to plot. If None, auto-select.
        time_labels: Labels for the time axis.
        save_path: Save path.
        title: Figure title.
    """
    T, H, W = predictions.shape

    if pixel_locations is None:
        # Auto-select: center + 3 random points with gaps
        pixel_locations = [(H // 2, W // 2)]
        gap_fraction = (1 - masks).mean(axis=0)
        # Find pixels with moderate gap fraction
        candidates = np.argwhere((gap_fraction > 0.1) & (gap_fraction < 0.9))
        if len(candidates) > 3:
            idx = np.random.choice(len(candidates), 3, replace=False)
            pixel_locations.extend([(int(r), int(c)) for r, c in candidates[idx]])
        elif len(candidates) > 0:
            pixel_locations.extend([(int(r), int(c)) for r, c in candidates])

    n_plots = len(pixel_locations)
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots), sharex=True)
    if n_plots == 1:
        axes = [axes]

    time_axis = time_labels if time_labels is not None else list(range(T))

    for ax, (row, col) in zip(axes, pixel_locations):
        pred_ts = predictions[:, row, col]
        target_ts = targets[:, row, col]
        mask_ts = masks[:, row, col]

        ax.plot(time_axis, target_ts, "o-", color="#4CAF50", label="Ground Truth", markersize=3, linewidth=1)
        ax.plot(time_axis, pred_ts, "s-", color="#2196F3", label="Prediction", markersize=3, linewidth=1)

        # Highlight gap timesteps
        gap_times = np.where(mask_ts < 1)[0]
        if len(gap_times) > 0:
            for gt in gap_times:
                ax.axvspan(
                    max(0, gt - 0.4), min(len(time_axis) - 1, gt + 0.4),
                    alpha=0.1, color="red",
                )

        ax.set_ylabel("Value")
        ax.set_title(f"Pixel ({row}, {col})")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Time series plot saved to %s", save_path)
    return fig
