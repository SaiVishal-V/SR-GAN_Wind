"""
Classical gap-filling baselines for WindGapGAN.

Provides simple, non-learned baselines that serve as lower bounds
for gap-filling performance. These must be outperformed by any
deep learning model to justify its complexity.

Baselines:
    - Persistence: Use the last observed value in time.
    - Linear Interpolation: Interpolate spatially.
    - Nearest Neighbor: Use nearest valid pixel.
    - Mean Filling: Fill gaps with the spatial mean.

All baselines operate on numpy arrays and return filled arrays
with the same shape. Observed pixels are never modified.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


class PersistenceBaseline:
    """
    Fill gaps using the last observed timestep (persistence/last-value).

    For each missing pixel at time t, copy the value from time t-1.
    If t-1 is also missing, try t-2, etc. If no previous observation
    exists, try forward filling.
    """

    name = "Persistence"

    @staticmethod
    def fill(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Args:
            data: (T, H, W) array with observed values (NaN for missing).
            mask: (T, H, W) binary mask, 1=observed, 0=missing.

        Returns:
            Filled array of shape (T, H, W).
        """
        T, H, W = data.shape
        filled = data.copy()

        for t in range(1, T):
            # Where current time is missing, use previous time
            gap_pixels = mask[t] == 0
            filled[t][gap_pixels] = filled[t - 1][gap_pixels]

        # Forward fill first timestep if needed
        if (mask[0] == 0).any():
            for t in range(1, T):
                still_missing = np.isnan(filled[0]) | (mask[0] == 0)
                if not still_missing.any():
                    break
                filled[0][still_missing] = filled[t][still_missing]

        # Any remaining NaN → fill with global mean
        global_mean = np.nanmean(data)
        filled = np.nan_to_num(filled, nan=global_mean)

        return filled


class LinearInterpolationBaseline:
    """
    Fill gaps using spatial linear interpolation (griddata).

    For each timestep, interpolate missing pixels from observed
    neighbors using scipy's linear griddata.
    """

    name = "Linear Interpolation"

    @staticmethod
    def fill(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Args:
            data: (T, H, W) array.
            mask: (T, H, W) binary mask.

        Returns:
            Filled array of shape (T, H, W).
        """
        T, H, W = data.shape
        filled = data.copy()

        rows, cols = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")

        for t in range(T):
            observed = mask[t] > 0
            missing = mask[t] == 0

            if not missing.any() or not observed.any():
                continue

            obs_points = np.column_stack([rows[observed], cols[observed]])
            obs_values = data[t][observed]
            miss_points = np.column_stack([rows[missing], cols[missing]])

            try:
                interpolated = griddata(
                    obs_points, obs_values, miss_points,
                    method="linear", fill_value=np.nanmean(obs_values)
                )
                filled[t][missing] = interpolated
            except Exception:
                # Fallback to nearest
                interpolated = griddata(
                    obs_points, obs_values, miss_points,
                    method="nearest"
                )
                filled[t][missing] = interpolated

        filled = np.nan_to_num(filled, nan=float(np.nanmean(data)))
        return filled


class NearestNeighborBaseline:
    """
    Fill gaps with the nearest valid pixel value.

    Uses scipy.ndimage distance transform to find nearest neighbors.
    """

    name = "Nearest Neighbor"

    @staticmethod
    def fill(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Args:
            data: (T, H, W) array.
            mask: (T, H, W) binary mask.

        Returns:
            Filled array of shape (T, H, W).
        """
        T, H, W = data.shape
        filled = data.copy()

        for t in range(T):
            missing = mask[t] == 0
            if not missing.any():
                continue

            observed = mask[t] > 0
            if not observed.any():
                filled[t] = np.nanmean(data)
                continue

            # Use distance transform to find nearest observed pixel
            _, indices = ndimage.distance_transform_edt(
                ~observed, return_distances=True, return_indices=True
            )
            filled[t] = data[t][indices[0], indices[1]]

        filled = np.nan_to_num(filled, nan=float(np.nanmean(data)))
        return filled


class MeanFillingBaseline:
    """
    Fill gaps with the spatial mean of observed pixels per timestep.

    This is the simplest possible baseline.
    """

    name = "Mean Filling"

    @staticmethod
    def fill(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Args:
            data: (T, H, W) array.
            mask: (T, H, W) binary mask.

        Returns:
            Filled array of shape (T, H, W).
        """
        T, H, W = data.shape
        filled = data.copy()

        for t in range(T):
            observed = mask[t] > 0
            missing = mask[t] == 0

            if not missing.any():
                continue

            if observed.any():
                spatial_mean = float(np.nanmean(data[t][observed]))
            else:
                spatial_mean = float(np.nanmean(data))

            filled[t][missing] = spatial_mean

        filled = np.nan_to_num(filled, nan=float(np.nanmean(data)))
        return filled


# ── Registry ───────────────────────────────────────────────────────────────

CLASSICAL_BASELINES = {
    "persistence": PersistenceBaseline,
    "linear_interpolation": LinearInterpolationBaseline,
    "nearest_neighbor": NearestNeighborBaseline,
    "mean_filling": MeanFillingBaseline,
}


def run_all_baselines(
    data: np.ndarray, mask: np.ndarray
) -> dict[str, np.ndarray]:
    """
    Run all classical baselines and return their predictions.

    Args:
        data: (T, H, W) array with NaN for missing.
        mask: (T, H, W) binary mask.

    Returns:
        Dict mapping baseline name to filled array.
    """
    results = {}
    for name, baseline_cls in CLASSICAL_BASELINES.items():
        logger.info("Running baseline: %s", baseline_cls.name)
        results[name] = baseline_cls.fill(data, mask)
    return results
