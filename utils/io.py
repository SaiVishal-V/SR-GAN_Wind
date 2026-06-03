"""
NetCDF I/O utilities for WindGapGAN.

Provides functions for reading, writing, and inspecting NetCDF files
with xarray, ensuring compatibility with Panoply, xarray, and netCDF4.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


def load_dataset(nc_path: str | Path) -> xr.Dataset:
    """
    Load a NetCDF dataset with error handling.

    Args:
        nc_path: Path to the .nc file.

    Returns:
        Loaded xarray Dataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be read as NetCDF.
    """
    nc_path = Path(nc_path)
    if not nc_path.exists():
        raise FileNotFoundError(f"NetCDF file not found: {nc_path}")
    if nc_path.suffix not in (".nc", ".nc4", ".hdf5", ".h5"):
        logger.warning("File extension '%s' is unusual for NetCDF. Attempting to read anyway.", nc_path.suffix)

    try:
        ds = xr.open_dataset(nc_path, engine="netcdf4")
    except Exception as e:
        raise ValueError(f"Failed to open NetCDF file '{nc_path}': {e}") from e

    logger.info("Loaded dataset from %s (%d variables, %d dimensions)", nc_path, len(ds.data_vars), len(ds.dims))
    return ds


def save_predictions_netcdf(
    save_path: str | Path,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    observed_mask: np.ndarray,
    time_coords: np.ndarray,
    lat_coords: np.ndarray,
    lon_coords: np.ndarray,
    variable_name: str = "wind_speed",
    attrs: Optional[dict] = None,
) -> None:
    """
    Save gap-filling predictions as a NetCDF file.

    The output file contains:
        - {variable_name}_predicted: Model predictions
        - {variable_name}_ground_truth: Original ground truth
        - {variable_name}_error: Prediction - Ground Truth
        - observed_mask: Binary mask (1=observed, 0=missing)

    Args:
        save_path: Output file path.
        predictions: Predicted field, shape (T, H, W).
        ground_truth: Ground truth field, shape (T, H, W).
        observed_mask: Observation mask, shape (T, H, W).
        time_coords: Time coordinate values.
        lat_coords: Latitude coordinate values.
        lon_coords: Longitude coordinate values.
        variable_name: Name of the target variable.
        attrs: Optional global attributes.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    error = predictions - ground_truth

    ds = xr.Dataset(
        {
            f"{variable_name}_predicted": (
                ["time", "latitude", "longitude"],
                predictions.astype(np.float32),
            ),
            f"{variable_name}_ground_truth": (
                ["time", "latitude", "longitude"],
                ground_truth.astype(np.float32),
            ),
            f"{variable_name}_error": (
                ["time", "latitude", "longitude"],
                error.astype(np.float32),
            ),
            "observed_mask": (
                ["time", "latitude", "longitude"],
                observed_mask.astype(np.int8),
            ),
        },
        coords={
            "time": time_coords,
            "latitude": lat_coords,
            "longitude": lon_coords,
        },
    )

    # Add metadata
    ds[f"{variable_name}_predicted"].attrs["long_name"] = f"Gap-filled {variable_name}"
    ds[f"{variable_name}_ground_truth"].attrs["long_name"] = f"Original {variable_name}"
    ds[f"{variable_name}_error"].attrs["long_name"] = f"Prediction error ({variable_name})"
    ds["observed_mask"].attrs["long_name"] = "Observation mask (1=observed, 0=gap)"

    if attrs:
        ds.attrs.update(attrs)
    ds.attrs["created_by"] = "WindGapGAN"
    ds.attrs["description"] = "Gap-filling predictions from WindGapGAN framework"

    ds.to_netcdf(save_path, engine="netcdf4")
    logger.info("Predictions saved to %s", save_path)


def get_dataset_summary(ds: xr.Dataset) -> str:
    """
    Generate a human-readable summary of a NetCDF dataset.

    Args:
        ds: xarray Dataset.

    Returns:
        Formatted string summary.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("DATASET SUMMARY")
    lines.append("=" * 60)

    lines.append(f"\nDimensions ({len(ds.dims)}):")
    for dim_name, dim_size in ds.dims.items():
        lines.append(f"  {dim_name}: {dim_size}")

    lines.append(f"\nCoordinates ({len(ds.coords)}):")
    for coord_name, coord in ds.coords.items():
        lines.append(f"  {coord_name}: {coord.dtype}, shape={coord.shape}")

    lines.append(f"\nData Variables ({len(ds.data_vars)}):")
    for var_name, var in ds.data_vars.items():
        nan_count = int(np.isnan(var.values).sum()) if np.issubdtype(var.dtype, np.floating) else 0
        total = int(np.prod(var.shape))
        nan_pct = (nan_count / total * 100) if total > 0 else 0
        lines.append(f"  {var_name}: {var.dtype}, shape={var.shape}, NaN={nan_pct:.1f}%")

    lines.append(f"\nGlobal Attributes ({len(ds.attrs)}):")
    for attr_name, attr_val in list(ds.attrs.items())[:10]:
        lines.append(f"  {attr_name}: {attr_val}")
    if len(ds.attrs) > 10:
        lines.append(f"  ... and {len(ds.attrs) - 10} more")

    lines.append("=" * 60)
    return "\n".join(lines)
