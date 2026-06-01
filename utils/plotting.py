"""
Plotting utilities for wind-speed super-resolution evaluation.

Rules (from project plan):
    - matplotlib ONLY (no cartopy, no basemap)
    - Wind-speed colormap: vmin=0, vmax=30
    - Missing observations: white (cmap.set_bad("white"))
    - Denormalize for visualization
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Optional

# Verified normalization constants
NORM_MEAN = 6.336302810300043
NORM_STD = 2.9275431488456434


def denormalize_np(x: np.ndarray) -> np.ndarray:
    """Denormalize wind speed from z-score to m/s."""
    return x * NORM_STD + NORM_MEAN


def _prepare_wind_cmap(cmap_name: str = "viridis") -> matplotlib.colors.Colormap:
    """Create colormap with white for missing/masked values."""
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("white")
    return cmap


def plot_wind_field(
    data: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    vmin: float = 0,
    vmax: float = 30,
    cmap_name: str = "viridis",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    is_normalized: bool = True,
) -> None:
    """
    Plot a single wind-speed field with proper masking.

    Args:
        data: 2D wind-speed array (H, W).
        mask: 2D ocean mask (H, W), 1=ocean, 0=land.
        title: Plot title.
        vmin: Colorbar minimum (m/s).
        vmax: Colorbar maximum (m/s).
        cmap_name: Colormap name.
        save_path: If provided, save figure to this path.
        figsize: Figure size.
        is_normalized: If True, denormalize before plotting.
    """
    if is_normalized:
        data = denormalize_np(data)

    cmap = _prepare_wind_cmap(cmap_name)

    # Mask land pixels as NaN for white display
    data_masked = np.where(mask == 1, data, np.nan)

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    im = ax.imshow(
        data_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto",
        extent=[30, 100, -10, 30]
    )
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    plt.colorbar(im, ax=ax, label="Wind Speed (m/s)", shrink=0.8)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_comparison(
    lr: np.ndarray,
    sr: np.ndarray,
    hr: np.ndarray,
    mask_hr: np.ndarray,
    mask_lr: Optional[np.ndarray] = None,
    vmin: float = 0,
    vmax: float = 30,
    cmap_name: str = "viridis",
    save_path: Optional[str] = None,
    title: str = "",
    is_normalized: bool = True,
) -> None:
    """
    Plot LR input, SR prediction, ground truth, and error map side by side.

    Args:
        lr: LR input (H_lr, W_lr).
        sr: SR prediction (H_hr, W_hr).
        hr: HR ground truth (H_hr, W_hr).
        mask_hr: HR ocean mask (H_hr, W_hr).
        mask_lr: LR ocean mask (optional).
        vmin: Colorbar minimum (m/s).
        vmax: Colorbar maximum (m/s).
        cmap_name: Colormap name.
        save_path: Save path.
        title: Super-title.
        is_normalized: If True, denormalize before plotting.
    """
    if is_normalized:
        lr = denormalize_np(lr)
        sr = denormalize_np(sr)
        hr = denormalize_np(hr)

    cmap = _prepare_wind_cmap(cmap_name)
    error_cmap = _prepare_wind_cmap("RdBu_r")

    # Mask land
    sr_masked = np.where(mask_hr == 1, sr, np.nan)
    hr_masked = np.where(mask_hr == 1, hr, np.nan)
    error = np.where(mask_hr == 1, sr - hr, np.nan)

    if mask_lr is not None:
        lr_masked = np.where(mask_lr == 1, lr, np.nan)
    else:
        lr_masked = lr

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Extent for geographic mapping (lon_min, lon_max, lat_min, lat_max)
    geo_extent = [30, 100, -10, 30]

    # LR Input
    im0 = axes[0].imshow(lr_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", extent=geo_extent)
    axes[0].set_title("LR Input", fontsize=12)
    axes[0].set_xlabel("Longitude (°E)")
    axes[0].set_ylabel("Latitude (°N)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # SR Prediction
    im1 = axes[1].imshow(sr_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", extent=geo_extent)
    axes[1].set_title("SR Prediction", fontsize=12)
    axes[1].set_xlabel("Longitude (°E)")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # Ground Truth
    im2 = axes[2].imshow(hr_masked, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", extent=geo_extent)
    axes[2].set_title("Ground Truth (HR)", fontsize=12)
    axes[2].set_xlabel("Longitude (°E)")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    # Error Map
    error_max = max(abs(np.nanmin(error)), abs(np.nanmax(error)), 3.0)
    im3 = axes[3].imshow(
        error, cmap=error_cmap, vmin=-error_max, vmax=error_max, aspect="auto", extent=geo_extent
    )
    axes[3].set_title("Error (SR - HR)", fontsize=12)
    axes[3].set_xlabel("Longitude (°E)")
    plt.colorbar(im3, ax=axes[3], label="Error (m/s)", shrink=0.8)

    if title:
        fig.suptitle(title, fontsize=14, y=1.02)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_scatter(
    pred_values: np.ndarray,
    target_values: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Prediction vs Ground Truth",
) -> None:
    """
    Scatter plot of predicted vs ground truth wind speeds (m/s).

    Args:
        pred_values: 1D array of predicted values (m/s).
        target_values: 1D array of ground truth values (m/s).
        save_path: Save path.
        title: Plot title.
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    # Subsample for large datasets
    n = len(pred_values)
    if n > 50000:
        idx = np.random.choice(n, 50000, replace=False)
        pred_values = pred_values[idx]
        target_values = target_values[idx]

    ax.scatter(target_values, pred_values, alpha=0.1, s=1, c="steelblue")

    # 1:1 line
    lims = [0, max(target_values.max(), pred_values.max()) + 1]
    ax.plot(lims, lims, "r--", linewidth=1.5, label="1:1 Line")

    ax.set_xlabel("Ground Truth (m/s)", fontsize=12)
    ax.set_ylabel("Prediction (m/s)", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_histogram(
    pred_values: np.ndarray,
    target_values: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Wind Speed Distribution",
    bins: int = 100,
) -> None:
    """
    Histogram comparing predicted and ground truth wind-speed distributions (m/s).

    Args:
        pred_values: 1D array of predicted values (m/s).
        target_values: 1D array of ground truth values (m/s).
        save_path: Save path.
        title: Plot title.
        bins: Number of histogram bins.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ax.hist(
        target_values, bins=bins, alpha=0.5, label="Ground Truth",
        density=True, color="steelblue",
    )
    ax.hist(
        pred_values, bins=bins, alpha=0.5, label="Prediction",
        density=True, color="coral",
    )

    ax.set_xlabel("Wind Speed (m/s)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    val_metrics: dict = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        train_losses: List of training losses per epoch.
        val_losses: List of validation losses per epoch.
        val_metrics: Optional dict of {metric_name: [values_per_epoch]}.
        save_path: Save path.
    """
    n_plots = 1 + (len(val_metrics) if val_metrics else 0)
    fig, axes = plt.subplots(1, min(n_plots, 4), figsize=(6 * min(n_plots, 4), 5))

    if n_plots == 1:
        axes = [axes]

    # Loss curve
    axes[0].plot(train_losses, label="Train Loss", color="steelblue")
    axes[0].plot(val_losses, label="Val Loss", color="coral")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Additional metrics
    if val_metrics:
        for i, (name, values) in enumerate(val_metrics.items()):
            if i + 1 >= len(axes):
                break
            axes[i + 1].plot(values, label=name, color="steelblue")
            axes[i + 1].set_xlabel("Epoch")
            axes[i + 1].set_ylabel(name)
            axes[i + 1].set_title(f"Validation {name}")
            axes[i + 1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
