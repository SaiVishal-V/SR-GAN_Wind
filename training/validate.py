"""
Validation module for wind-speed super-resolution.

Supports both:
    - Patch-based validation (matching training mode)
    - Full-scene validation (review #2: mandatory for geospatial SR)

Every validation epoch processes both patch metrics AND full-scene metrics
on complete (80,140) → (320,560) timesteps.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, Optional
from tqdm import tqdm

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import compute_all_metrics, masked_rmse, masked_mae


@torch.no_grad()
def validate_patches(
    generator: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validate on patches — computes average metrics across all validation patches.

    Args:
        generator: Generator model (eval mode).
        val_loader: Validation DataLoader (patch mode).
        device: Computation device.

    Returns:
        Dictionary of averaged metrics.
    """
    generator.eval()

    metrics_sum = {}
    n_batches = 0

    for batch in val_loader:
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        mask = batch["mask"].to(device)

        sr = generator(lr)

        # Compute metrics per batch
        batch_metrics = {
            "rmse": masked_rmse(sr, hr, mask),
            "mae": masked_mae(sr, hr, mask),
        }

        for key, val in batch_metrics.items():
            metrics_sum[key] = metrics_sum.get(key, 0.0) + val
        n_batches += 1

    # Average over batches
    avg_metrics = {k: v / max(n_batches, 1) for k, v in metrics_sum.items()}
    return avg_metrics


@torch.no_grad()
def validate_full_scenes(
    generator: torch.nn.Module,
    val_dataset,
    device: torch.device,
    num_scenes: Optional[int] = None,
) -> Dict[str, float]:
    """
    Validate on full scenes — processes entire (80,140) → (320,560) timesteps.
    (Review #2: mandatory for geospatial super-resolution)

    Args:
        generator: Generator model (eval mode).
        val_dataset: Validation dataset in 'full' mode.
        device: Computation device.
        num_scenes: Max number of scenes to evaluate (None = all).

    Returns:
        Dictionary of averaged full-scene metrics.
    """
    generator.eval()

    n_scenes = len(val_dataset)
    if num_scenes is not None:
        n_scenes = min(n_scenes, num_scenes)

    all_metrics = {}
    count = 0

    for i in range(n_scenes):
        sample = val_dataset[i]
        lr = sample["lr"].unsqueeze(0).to(device)
        hr = sample["hr"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)

        sr = generator(lr)

        scene_metrics = compute_all_metrics(sr, hr, mask)

        for key, val in scene_metrics.items():
            all_metrics[key] = all_metrics.get(key, 0.0) + val
        count += 1

    # Average over scenes
    avg_metrics = {}
    for key, val in all_metrics.items():
        avg_metrics[f"scene_{key}"] = val / max(count, 1)

    return avg_metrics


@torch.no_grad()
def validate(
    generator: torch.nn.Module,
    val_loader: DataLoader,
    val_full_dataset,
    device: torch.device,
    num_full_scenes: int = 5,
) -> Dict[str, float]:
    """
    Combined validation: patch metrics + full-scene metrics.

    Args:
        generator: Generator model.
        val_loader: Validation DataLoader (patch mode).
        val_full_dataset: Validation dataset in 'full' mode.
        device: Computation device.
        num_full_scenes: Number of full scenes to evaluate.

    Returns:
        Combined dictionary of all validation metrics.
    """
    # Patch-level metrics
    patch_metrics = validate_patches(generator, val_loader, device)

    # Full-scene metrics
    scene_metrics = validate_full_scenes(
        generator, val_full_dataset, device, num_scenes=num_full_scenes
    )

    # Combine
    all_metrics = {}
    all_metrics.update(patch_metrics)
    all_metrics.update(scene_metrics)

    return all_metrics
