"""
Loss and metric curve plotting for WindGapGAN.

Generates publication-quality training curves including:
    - Training/validation loss
    - RMSE, MAE, Bias, Correlation over epochs
    - Learning rate schedule
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# Consistent style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
})


def plot_loss_curves(
    train_losses: list[float],
    val_losses: list[float],
    save_path: Optional[str | Path] = None,
    title: str = "Training & Validation Loss",
) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Train Loss", color="#2196F3", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss", color="#F44336", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.set_yscale("log")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Loss curves saved to %s", save_path)
    return fig


def plot_metric_curves(
    metrics_history: dict[str, list[float]],
    save_path: Optional[str | Path] = None,
    title: str = "Evaluation Metrics Over Training",
) -> plt.Figure:
    """
    Plot multiple metric curves on subplots.

    Args:
        metrics_history: Dict mapping metric names to lists of values per epoch.
            Expected keys: 'rmse_gap', 'mae_gap', 'bias_gap', 'corr_gap'.
    """
    metric_configs = {
        "rmse_gap": {"label": "RMSE (Gap)", "color": "#E91E63", "ylabel": "RMSE"},
        "mae_gap": {"label": "MAE (Gap)", "color": "#9C27B0", "ylabel": "MAE"},
        "bias_gap": {"label": "Bias (Gap)", "color": "#FF9800", "ylabel": "Bias"},
        "corr_gap": {"label": "Correlation (Gap)", "color": "#4CAF50", "ylabel": "Correlation"},
    }

    available = [k for k in metric_configs if k in metrics_history]
    n_plots = len(available)
    if n_plots == 0:
        logger.warning("No metrics to plot.")
        return plt.figure()

    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    for ax, key in zip(axes, available):
        cfg = metric_configs[key]
        values = metrics_history[key]
        epochs = range(1, len(values) + 1)
        ax.plot(epochs, values, color=cfg["color"], linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(cfg["ylabel"])
        ax.set_title(cfg["label"])

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Metric curves saved to %s", save_path)
    return fig


def plot_learning_rate(
    lr_history: list[float],
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot learning rate schedule."""
    fig, ax = plt.subplots(figsize=(10, 4))
    epochs = range(1, len(lr_history) + 1)

    ax.plot(epochs, lr_history, color="#607D8B", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_yscale("log")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
