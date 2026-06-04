"""
Core NetCDF dataset for WindGapGAN.

Loads any NetCDF file, extracts temporal sequences with spatial patches,
applies adaptive normalization, and generates training samples with
synthetic masks for gap-filling supervision.

Key design decisions:
    - Ground truth preservation: stores original complete field.
    - Synthetic gaps: during training, artificial masks create gaps
      so the model learns to reconstruct from context.
    - Temporal windows: contiguous T-step windows for sequence modeling.
    - Spatial patches: configurable crop size for memory management.
    - Strict temporal splitting: train/val/test by time, no leakage.
    - Adaptive normalization: auto-selects best method for the data.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import xarray as xr
from scipy import stats
from torch.utils.data import Dataset

from datasets.mask_generator import MaskGenerator
from datasets.variable_discovery import (
    detect_mask_variable,
    detect_missing_values,
    discover_dimensions,
    discover_variables,
    prompt_variable_selection,
)

logger = logging.getLogger(__name__)


# ── Normalization Utilities ────────────────────────────────────────────────


def compute_normalization_stats(
    data: np.ndarray, method: str = "auto"
) -> dict[str, Any]:
    """
    Compute normalization statistics for a dataset.

    Methods:
        - auto: Analyze distribution and pick best method.
        - zscore: (x - mean) / std
        - minmax: (x - min) / (max - min)
        - robust: (x - median) / IQR
        - none: No normalization

    Args:
        data: Array of valid (non-NaN) values.
        method: Normalization method.

    Returns:
        Dict with 'method' and method-specific parameters.
    """
    valid = data[~np.isnan(data)].astype(np.float64)
    if len(valid) == 0:
        logger.warning("No valid values for normalization. Using 'none'.")
        return {"method": "none"}

    if method == "auto":
        # Heuristic: check skewness to decide
        skewness = float(stats.skew(valid))
        kurtosis = float(stats.kurtosis(valid))
        logger.info(
            "Auto-normalization analysis: skewness=%.3f, kurtosis=%.3f",
            skewness,
            kurtosis,
        )

        if abs(skewness) > 2.0:
            # Highly skewed → robust normalization
            method = "robust"
        elif abs(skewness) > 1.0:
            # Moderately skewed → min-max
            method = "minmax"
        else:
            # Roughly symmetric → z-score
            method = "zscore"

        logger.info("Auto-selected normalization method: '%s'", method)

    if method == "zscore":
        return {
            "method": "zscore",
            "mean": float(np.mean(valid)),
            "std": float(np.std(valid)),
        }
    elif method == "minmax":
        return {
            "method": "minmax",
            "min": float(np.min(valid)),
            "max": float(np.max(valid)),
        }
    elif method == "robust":
        q25 = float(np.percentile(valid, 25))
        q75 = float(np.percentile(valid, 75))
        iqr = q75 - q25
        return {
            "method": "robust",
            "median": float(np.median(valid)),
            "iqr": iqr if iqr > 0 else 1.0,
        }
    elif method == "none":
        return {"method": "none"}
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")


def normalize(data: np.ndarray, norm_stats: dict[str, Any]) -> np.ndarray:
    """Apply normalization using precomputed statistics."""
    method = norm_stats["method"]
    result = data.copy()

    if method == "zscore":
        std = norm_stats["std"]
        if std < 1e-8:
            std = 1.0
        result = (result - norm_stats["mean"]) / std
    elif method == "minmax":
        range_val = norm_stats["max"] - norm_stats["min"]
        if range_val < 1e-8:
            range_val = 1.0
        result = (result - norm_stats["min"]) / range_val
    elif method == "robust":
        result = (result - norm_stats["median"]) / norm_stats["iqr"]
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization method: '{method}'")

    return result


def denormalize(data: np.ndarray, norm_stats: dict[str, Any]) -> np.ndarray:
    """Reverse normalization."""
    method = norm_stats["method"]
    result = data.copy()

    if method == "zscore":
        std = norm_stats["std"]
        if std < 1e-8:
            std = 1.0
        result = result * std + norm_stats["mean"]
    elif method == "minmax":
        range_val = norm_stats["max"] - norm_stats["min"]
        if range_val < 1e-8:
            range_val = 1.0
        result = result * range_val + norm_stats["min"]
    elif method == "robust":
        result = result * norm_stats["iqr"] + norm_stats["median"]
    elif method == "none":
        pass

    return result


# ── Dataset Class ──────────────────────────────────────────────────────────


class WindGapDataset(Dataset):
    """
    PyTorch Dataset for spatio-temporal gap filling from NetCDF data.

    Produces samples of shape:
        input:  (T, 2, H, W)  — [normalized_field, observed_mask]
        target: (T, 1, H, W)  — [ground_truth_normalized_field]
        mask:   (T, 1, H, W)  — [observed_mask: 1=valid, 0=gap]

    During training, synthetic masks are applied to create artificial gaps.
    During validation/test, real missing patterns are used.

    Performance design:
        - Stochastic sampling: each epoch uses a fixed random sample budget
          (max_samples_per_epoch) instead of exhaustively iterating all patches.
        - Lazy index generation: patch indices (t, r, c) are randomly generated
          in reshuffle(), not pre-enumerated.
        - Pre-generated masks: synthetic masks are created once per epoch in
          reshuffle(), not per-sample in __getitem__.
        - Data cached as tensors in RAM for zero-copy slicing.
    """

    def __init__(
        self,
        data: np.ndarray,
        real_mask: np.ndarray,
        split: str = "train",
        sequence_length: int = 1,
        patch_size: int = 64,
        stride: int = 32,
        norm_stats: Optional[dict[str, Any]] = None,
        mask_generator: Optional[MaskGenerator] = None,
        augment: bool = False,
        max_samples_per_epoch: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        super().__init__()

        self.split = split
        self.sequence_length = sequence_length
        self.patch_size = patch_size
        self.stride = stride
        self.norm_stats = norm_stats
        self.mask_generator = mask_generator
        self.augment = augment and (split == "train")
        self.max_samples_per_epoch = max_samples_per_epoch
        self.rng = np.random.RandomState(seed)

        # Replace NaN with 0 in data (masked regions)
        data = np.nan_to_num(data.astype(np.float32), nan=0.0)

        # Normalize
        if norm_stats and norm_stats["method"] != "none":
            data = normalize(data, norm_stats)

        # Store as PyTorch tensors — cached in RAM for zero-copy slicing
        self.data = torch.from_numpy(data)
        self.real_mask = torch.from_numpy(real_mask.astype(np.float32))

        # Spatial/temporal bounds for random sampling
        T, H, W = self.data.shape
        self._T = T
        self._H = H
        self._W = W
        self._max_t = max(1, T - sequence_length + 1)
        self._max_r = max(1, H - patch_size + 1)
        self._max_c = max(1, W - patch_size + 1)

        # Compute full dataset size (for reference / val/test)
        n_t = self._max_t
        n_r = len(range(0, H - patch_size + 1, stride))
        n_c = len(range(0, W - patch_size + 1, stride))
        self._full_dataset_size = n_t * n_r * n_c

        # Determine effective epoch size
        if max_samples_per_epoch is not None and split == "train":
            self._epoch_size = min(max_samples_per_epoch, self._full_dataset_size)
        elif max_samples_per_epoch is not None and split == "val":
            # Validation uses a smaller budget
            self._epoch_size = min(max_samples_per_epoch // 4, self._full_dataset_size)
        else:
            self._epoch_size = self._full_dataset_size

        # Storage for current epoch's samples and pre-generated masks
        self.samples: list[tuple[int, int, int]] = []
        self._cached_masks: Optional[list[torch.Tensor]] = None

        # Initial population
        self.reshuffle()

        logger.info(
            "WindGapDataset[%s]: %d samples/epoch (full=%d, T=%d, patch=%d, data=%s)",
            split,
            self._epoch_size,
            self._full_dataset_size,
            sequence_length,
            patch_size,
            list(self.data.shape),
        )

    def reshuffle(self) -> None:
        """
        Re-generate random patch indices and pre-generate masks for the epoch.

        Called at the start of each training epoch by the trainer.
        For val/test with no max_samples_per_epoch, builds the deterministic grid.
        """
        if self.max_samples_per_epoch is not None:
            # Stochastic sampling: randomly generate (t, r, c) tuples
            ts = self.rng.randint(0, self._max_t, size=self._epoch_size)
            rs = self.rng.randint(0, self._max_r, size=self._epoch_size)
            cs = self.rng.randint(0, self._max_c, size=self._epoch_size)
            self.samples = list(zip(ts.tolist(), rs.tolist(), cs.tolist()))
        else:
            # Exhaustive grid (val/test default)
            self.samples = []
            for t in range(self._max_t):
                for r in range(0, self._H - self.patch_size + 1, self.stride):
                    for c in range(0, self._W - self.patch_size + 1, self.stride):
                        self.samples.append((t, r, c))

        # Pre-generate synthetic masks for the epoch (training only)
        if self.split == "train" and self.mask_generator is not None:
            self._cached_masks = []
            for _ in range(len(self.samples)):
                mask_np = self.mask_generator.generate_sequence(
                    self.sequence_length, self.patch_size, self.patch_size
                )
                self._cached_masks.append(torch.from_numpy(mask_np))
        else:
            self._cached_masks = None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Dict with keys:
                'input':  (T, 2, H, W) — [field, mask]
                'target': (T, 1, H, W) — [ground_truth]
                'mask':   (T, 1, H, W) — [observation mask]
        """
        t_start, r_start, c_start = self.samples[idx]
        t_end = t_start + self.sequence_length
        r_end = r_start + self.patch_size
        c_end = c_start + self.patch_size

        # Extract patch (slice, no copy needed since we don't modify inplace)
        data_patch = self.data[t_start:t_end, r_start:r_end, c_start:c_end]
        mask_patch = self.real_mask[t_start:t_end, r_start:r_end, c_start:c_end]

        if self.split == "train" and self._cached_masks is not None:
            # Use pre-generated mask from reshuffle()
            synthetic_mask = self._cached_masks[idx]

            # Training mask: intersection of real observed and synthetic mask
            train_mask = mask_patch * synthetic_mask
            input_field = data_patch * train_mask
            target_field = data_patch
            used_mask = train_mask
        else:
            # Validation/test: use real mask only
            input_field = data_patch * mask_patch
            target_field = data_patch
            used_mask = mask_patch

        # Data augmentation (training only)
        if self.augment:
            input_field, target_field, used_mask = self._augment(
                input_field, target_field, used_mask
            )

        # Shape: (T, H, W) → (T, 1, H, W)
        input_field = input_field.unsqueeze(1)
        target_field = target_field.unsqueeze(1)
        used_mask = used_mask.unsqueeze(1)

        # Concatenate field + mask for input: (T, 2, H, W)
        input_tensor = torch.cat([input_field, used_mask], dim=1)

        return {
            "input": input_tensor,
            "target": target_field,
            "mask": used_mask,
        }

    def _augment(
        self,
        field: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply spatial augmentation (flips and rotations)."""
        # Random horizontal flip
        if np.random.random() < 0.5:
            field = torch.flip(field, dims=[-1])
            target = torch.flip(target, dims=[-1])
            mask = torch.flip(mask, dims=[-1])

        # Random vertical flip
        if np.random.random() < 0.5:
            field = torch.flip(field, dims=[-2])
            target = torch.flip(target, dims=[-2])
            mask = torch.flip(mask, dims=[-2])

        return field, target, mask


# ── Dataset Builder ────────────────────────────────────────────────────────


def build_datasets(
    nc_path: str | Path,
    target_variable: Optional[str] = None,
    sequence_length: int = 1,
    patch_size: int = 64,
    stride: int = 32,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    norm_method: str = "auto",
    mask_variable: Optional[str] = None,
    missing_values: Optional[list[float]] = None,
    synthetic_mask_strategy: str = "mixed",
    synthetic_mask_ratio: float = 0.3,
    max_samples_per_epoch: Optional[int] = 4800,
    seed: int = 42,
) -> tuple[WindGapDataset, WindGapDataset, WindGapDataset, dict[str, Any]]:
    """
    Build train/val/test datasets from a NetCDF file.

    This is the main entry point for dataset construction. It:
        1. Loads the NetCDF file.
        2. Discovers variables and dimensions.
        3. Detects missing values and builds observation mask.
        4. Performs temporal splitting (strict, no leakage).
        5. Computes normalization statistics from training data only.
        6. Creates dataset objects for each split.

    Args:
        nc_path: Path to NetCDF file.
        target_variable: Target variable name (None = interactive selection).
        sequence_length: Temporal window size.
        patch_size: Spatial patch size.
        stride: Patch extraction stride.
        train_ratio: Fraction of timesteps for training.
        val_ratio: Fraction of timesteps for validation.
        test_ratio: Fraction of timesteps for testing.
        norm_method: Normalization method.
        mask_variable: Explicit mask variable name.
        missing_values: User-defined sentinel values.
        synthetic_mask_strategy: Strategy for synthetic masks.
        synthetic_mask_ratio: Fraction of pixels to mask.
        max_samples_per_epoch: Stochastic sample budget per epoch (None = full dataset).
        seed: Random seed.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, metadata).
    """
    from utils.io import load_dataset

    # Load dataset
    ds = load_dataset(nc_path)

    # Discover variables
    variables = discover_variables(ds)

    # Select target variable
    target_var = prompt_variable_selection(variables, auto_select=target_variable)

    # Discover dimensions
    dims = discover_dimensions(ds)

    # Get data array
    da = ds[target_var]

    # Detect or load observation mask
    mask_var = mask_variable or detect_mask_variable(ds)
    if mask_var and mask_var in ds.data_vars:
        logger.info("Using explicit mask variable: '%s'", mask_var)
        real_mask = ds[mask_var].values.astype(np.float32)
        # Ensure binary
        real_mask = (real_mask > 0).astype(np.float32)
        mask_metadata = {
            "detection_methods": [f"explicit_variable={mask_var}"],
            "total_pixels": int(np.prod(real_mask.shape)),
            "total_missing": int((real_mask == 0).sum()),
            "missing_fraction": float((real_mask == 0).sum() / np.prod(real_mask.shape)),
            "observation_fraction": float((real_mask > 0).sum() / np.prod(real_mask.shape)),
        }
    else:
        real_mask, mask_metadata = detect_missing_values(da, user_sentinels=missing_values)

    # Extract data as numpy array: (time, lat, lon)
    data = da.values.astype(np.float32)

    # Validate data shape — must be 3D (T, H, W)
    if data.ndim != 3:
        raise ValueError(
            f"Expected 3D data (time, lat, lon), got shape {data.shape} "
            f"with dims {da.dims}. Ensure the data has time, lat, lon dimensions."
        )

    T, H, W = data.shape
    logger.info("Data shape: T=%d, H=%d, W=%d", T, H, W)

    # Broadcast mask to 3D if it's 2D (static mask)
    if real_mask.ndim == 2:
        if real_mask.shape != (H, W):
            raise ValueError(f"2D mask shape {real_mask.shape} does not match data spatial shape {(H, W)}")
        real_mask = np.broadcast_to(real_mask, (T, H, W)).astype(np.float32)
        logger.info("Broadcasted 2D static mask to 3D shape %s", real_mask.shape)
    elif real_mask.shape != (T, H, W):
        raise ValueError(f"Mask shape {real_mask.shape} does not match data shape {(T, H, W)}")

    # ── Strict temporal splitting ──────────────────────────────────────
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    train_end = int(T * train_ratio)
    val_end = int(T * (train_ratio + val_ratio))

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    train_mask = real_mask[:train_end]
    val_mask = real_mask[train_end:val_end]
    test_mask = real_mask[val_end:]

    logger.info(
        "Temporal split: train=%d, val=%d, test=%d timesteps",
        train_data.shape[0],
        val_data.shape[0],
        test_data.shape[0],
    )

    # ── Compute normalization from training data only ──────────────────
    train_valid = train_data[train_mask > 0]
    norm_stats = compute_normalization_stats(train_valid, method=norm_method)
    logger.info("Normalization: %s", norm_stats)

    # ── Create mask generator ──────────────────────────────────────────
    mask_gen = MaskGenerator(
        strategy=synthetic_mask_strategy,
        mask_ratio=synthetic_mask_ratio,
        seed=seed,
    )

    # ── Build datasets ─────────────────────────────────────────────────
    train_ds = WindGapDataset(
        data=train_data,
        real_mask=train_mask,
        split="train",
        sequence_length=sequence_length,
        patch_size=patch_size,
        stride=stride,
        norm_stats=norm_stats,
        mask_generator=mask_gen,
        augment=True,
        max_samples_per_epoch=max_samples_per_epoch,
        seed=seed,
    )

    val_ds = WindGapDataset(
        data=val_data,
        real_mask=val_mask,
        split="val",
        sequence_length=sequence_length,
        patch_size=patch_size,
        stride=stride,
        norm_stats=norm_stats,
        mask_generator=mask_gen,
        augment=False,
        max_samples_per_epoch=max_samples_per_epoch,
        seed=seed + 1,
    )

    test_ds = WindGapDataset(
        data=test_data,
        real_mask=test_mask,
        split="test",
        sequence_length=sequence_length,
        patch_size=patch_size,
        stride=stride,
        norm_stats=norm_stats,
        mask_generator=None,
        augment=False,
        max_samples_per_epoch=None,  # Test always uses full dataset
        seed=seed + 2,
    )

    # ── Metadata ───────────────────────────────────────────────────────
    metadata = {
        "target_variable": target_var,
        "dimensions": dims,
        "data_shape": {"T": T, "H": H, "W": W},
        "split_sizes": {
            "train": train_data.shape[0],
            "val": val_data.shape[0],
            "test": test_data.shape[0],
        },
        "norm_stats": norm_stats,
        "mask_metadata": mask_metadata,
        "time_coords": ds.coords[dims["time"]].values,
        "lat_coords": ds.coords[dims["lat"]].values,
        "lon_coords": ds.coords[dims["lon"]].values,
    }

    # Generate dataset report
    from datasets.variable_discovery import generate_dataset_report

    generate_dataset_report(
        ds=ds,
        target_variable=target_var,
        dimensions=dims,
        mask_metadata=mask_metadata,
        save_path=Path("reports") / "dataset_report.md",
    )

    ds.close()
    return train_ds, val_ds, test_ds, metadata
