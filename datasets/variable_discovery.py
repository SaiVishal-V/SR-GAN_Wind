"""
Dynamic NetCDF variable and dimension discovery for WindGapGAN.

Provides automatic detection of:
    - Data variables with shape/dtype information
    - Time, latitude, and longitude dimensions by name heuristics
    - Missing values from NaN, _FillValue, missing_value, and sentinel values
    - Dataset report generation for validation before training

No variable names are hardcoded. Everything is discovered at runtime.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

# ── Dimension Name Heuristics ──────────────────────────────────────────────

TIME_NAMES = {"time", "Time", "TIME", "T", "t", "date", "Date", "DATE", "datetime", "DATETIME"}
LAT_NAMES = {"lat", "latitude", "Lat", "Latitude", "LAT", "LATITUDE", "y", "Y", "rlat", "nav_lat"}
LON_NAMES = {"lon", "longitude", "Lon", "Longitude", "LON", "LONGITUDE", "x", "X", "rlon", "nav_lon"}

# ── Mask Variable Name Heuristics ──────────────────────────────────────────

MASK_NAMES = {
    "observed_mask",
    "cloud_mask",
    "valid_mask",
    "mask",
    "quality_flag",
    "qc_flag",
    "data_mask",
    "observation_mask",
}


def discover_variables(ds: xr.Dataset) -> dict[str, dict[str, Any]]:
    """
    Discover all data variables in a NetCDF dataset.

    Args:
        ds: xarray Dataset.

    Returns:
        Dictionary mapping variable names to their metadata:
            {
                'wind_speed': {
                    'shape': (365, 100, 200),
                    'dtype': dtype('float32'),
                    'dims': ('time', 'lat', 'lon'),
                    'nan_fraction': 0.15,
                    'attrs': {...}
                },
                ...
            }
    """
    variables = {}
    for name, var in ds.data_vars.items():
        total = int(np.prod(var.shape))
        if np.issubdtype(var.dtype, np.floating):
            nan_count = int(np.isnan(var.values).sum())
            nan_fraction = nan_count / total if total > 0 else 0.0
        else:
            nan_fraction = 0.0

        variables[name] = {
            "shape": var.shape,
            "dtype": var.dtype,
            "dims": var.dims,
            "nan_fraction": nan_fraction,
            "attrs": dict(var.attrs),
        }

    logger.info("Discovered %d data variables", len(variables))
    return variables


def discover_dimensions(ds: xr.Dataset) -> dict[str, Optional[str]]:
    """
    Auto-detect time, latitude, and longitude dimensions by name heuristics.

    Args:
        ds: xarray Dataset.

    Returns:
        Dictionary with keys 'time', 'lat', 'lon' mapped to detected
        dimension names, or None if not found.

    Raises:
        ValueError: If critical dimensions cannot be detected.
    """
    detected = {"time": None, "lat": None, "lon": None}

    all_dims = set(ds.dims.keys())
    all_coords = set(ds.coords.keys())
    search_space = all_dims | all_coords

    # Detect time
    for name in search_space:
        if name in TIME_NAMES:
            detected["time"] = name
            break
    if detected["time"] is None:
        # Try matching by 'units' attribute (e.g., 'days since ...')
        for name in search_space:
            coord = ds.coords.get(name) or ds.get(name)
            if coord is not None:
                units = coord.attrs.get("units", "")
                if "since" in str(units).lower():
                    detected["time"] = name
                    break

    # Detect latitude
    for name in search_space:
        if name in LAT_NAMES:
            detected["lat"] = name
            break
    if detected["lat"] is None:
        for name in search_space:
            coord = ds.coords.get(name) or ds.get(name)
            if coord is not None:
                units = str(coord.attrs.get("units", "")).lower()
                if "degrees_north" in units or "degree_north" in units:
                    detected["lat"] = name
                    break

    # Detect longitude
    for name in search_space:
        if name in LON_NAMES:
            detected["lon"] = name
            break
    if detected["lon"] is None:
        for name in search_space:
            coord = ds.coords.get(name) or ds.get(name)
            if coord is not None:
                units = str(coord.attrs.get("units", "")).lower()
                if "degrees_east" in units or "degree_east" in units:
                    detected["lon"] = name
                    break

    logger.info("Detected dimensions: time=%s, lat=%s, lon=%s", detected["time"], detected["lat"], detected["lon"])

    # Validate
    missing = [k for k, v in detected.items() if v is None]
    if missing:
        available = list(search_space)
        raise ValueError(
            f"Could not auto-detect dimensions: {missing}. "
            f"Available names: {available}. "
            f"Please specify them manually in the config."
        )

    return detected


def detect_missing_values(
    da: xr.DataArray,
    user_sentinels: Optional[list[float]] = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Detect missing values in a DataArray and generate an observation mask.

    Detection priority:
        1. NaN values
        2. _FillValue attribute
        3. missing_value attribute
        4. Known sentinel values (-9999, 9999)
        5. User-defined sentinel values

    Args:
        da: xarray DataArray to analyze.
        user_sentinels: Optional list of user-defined missing value sentinels.

    Returns:
        Tuple of:
            - observed_mask: np.ndarray of shape da.shape, dtype=float32.
              1.0 = observed, 0.0 = missing.
            - metadata: dict with detection details.
    """
    values = da.values.astype(np.float32)
    mask = np.ones_like(values, dtype=np.float32)
    metadata: dict[str, Any] = {"detection_methods": [], "total_missing": 0, "total_pixels": int(np.prod(values.shape))}

    # 1. NaN detection
    nan_mask = np.isnan(values)
    if nan_mask.any():
        mask[nan_mask] = 0.0
        metadata["detection_methods"].append(f"NaN ({int(nan_mask.sum())} pixels)")

    # 2. _FillValue
    fill_value = da.attrs.get("_FillValue", da.encoding.get("_FillValue", None))
    if fill_value is not None:
        fill_mask = np.isclose(values, float(fill_value), equal_nan=False)
        if fill_mask.any():
            mask[fill_mask] = 0.0
            metadata["detection_methods"].append(f"_FillValue={fill_value} ({int(fill_mask.sum())} pixels)")

    # 3. missing_value
    missing_val = da.attrs.get("missing_value", None)
    if missing_val is not None:
        miss_mask = np.isclose(values, float(missing_val), equal_nan=False)
        if miss_mask.any():
            mask[miss_mask] = 0.0
            metadata["detection_methods"].append(f"missing_value={missing_val} ({int(miss_mask.sum())} pixels)")

    # 4. Known sentinels
    known_sentinels = [-9999.0, 9999.0, -999.0, 999.0, -1e30, 1e30]
    for sentinel in known_sentinels:
        sent_mask = np.isclose(values, sentinel, equal_nan=False)
        if sent_mask.any():
            mask[sent_mask] = 0.0
            metadata["detection_methods"].append(f"sentinel={sentinel} ({int(sent_mask.sum())} pixels)")

    # 5. User-defined sentinels
    if user_sentinels:
        for sentinel in user_sentinels:
            sent_mask = np.isclose(values, float(sentinel), equal_nan=False)
            if sent_mask.any():
                mask[sent_mask] = 0.0
                metadata["detection_methods"].append(f"user_sentinel={sentinel} ({int(sent_mask.sum())} pixels)")

    total_missing = int((mask == 0).sum())
    metadata["total_missing"] = total_missing
    metadata["missing_fraction"] = total_missing / metadata["total_pixels"] if metadata["total_pixels"] > 0 else 0
    metadata["observation_fraction"] = 1.0 - metadata["missing_fraction"]

    logger.info(
        "Missing value detection: %.1f%% missing (%d/%d pixels). Methods: %s",
        metadata["missing_fraction"] * 100,
        total_missing,
        metadata["total_pixels"],
        ", ".join(metadata["detection_methods"]) if metadata["detection_methods"] else "none",
    )

    return mask, metadata


def detect_mask_variable(ds: xr.Dataset) -> Optional[str]:
    """
    Attempt to find an explicit mask variable in the dataset.

    Args:
        ds: xarray Dataset.

    Returns:
        Name of the mask variable if found, else None.
    """
    for name in ds.data_vars:
        if name.lower() in MASK_NAMES or any(mn in name.lower() for mn in MASK_NAMES):
            logger.info("Found explicit mask variable: '%s'", name)
            return name
    return None


def prompt_variable_selection(variables: dict[str, dict[str, Any]], auto_select: Optional[str] = None) -> str:
    """
    Interactive CLI variable selection.

    Displays available variables and prompts the user to select one.
    In non-interactive mode (e.g., Colab jobs), use auto_select.

    Args:
        variables: Output from discover_variables().
        auto_select: If provided, skip interactive prompt and use this variable.

    Returns:
        Selected variable name.

    Raises:
        ValueError: If selected variable is not in the dataset.
    """
    if auto_select:
        if auto_select not in variables:
            raise ValueError(
                f"Target variable '{auto_select}' not found. "
                f"Available: {list(variables.keys())}"
            )
        logger.info("Auto-selected target variable: '%s'", auto_select)
        return auto_select

    print("\n" + "=" * 50)
    print("AVAILABLE DATA VARIABLES")
    print("=" * 50)
    var_list = list(variables.keys())
    for i, (name, info) in enumerate(variables.items(), 1):
        shape_str = "×".join(str(s) for s in info["shape"])
        nan_str = f"{info['nan_fraction'] * 100:.1f}% missing" if info["nan_fraction"] > 0 else "complete"
        print(f"  {i}. {name:<30s} {info['dtype']!s:<12s} ({shape_str}) [{nan_str}]")
    print("=" * 50)

    while True:
        try:
            selection = input("\nEnter target variable name or number: ").strip()
            if selection.isdigit():
                idx = int(selection) - 1
                if 0 <= idx < len(var_list):
                    selected = var_list[idx]
                    break
                else:
                    print(f"Invalid number. Enter 1–{len(var_list)}.")
            elif selection in variables:
                selected = selection
                break
            else:
                print(f"Variable '{selection}' not found. Try again.")
        except (EOFError, KeyboardInterrupt):
            raise ValueError("No variable selected. Aborting.")

    logger.info("User selected target variable: '%s'", selected)
    return selected


def generate_dataset_report(
    ds: xr.Dataset,
    target_variable: str,
    dimensions: dict[str, str],
    mask_metadata: dict[str, Any],
    save_path: Optional[str | Path] = None,
) -> str:
    """
    Generate a comprehensive dataset validation report.

    Training stops if:
        - No time dimension exists
        - No spatial dimensions exist
        - Missing value detection fails completely

    Args:
        ds: xarray Dataset.
        target_variable: Selected target variable name.
        dimensions: Detected dimensions dict.
        mask_metadata: Output from detect_missing_values().
        save_path: If provided, write report to this file.

    Returns:
        Report as a formatted string.
    """
    da = ds[target_variable]
    lines = []

    lines.append("# Dataset Validation Report")
    lines.append(f"\n**Generated by WindGapGAN**\n")

    # Variables
    lines.append("## Variables")
    for name, var in ds.data_vars.items():
        marker = " ← TARGET" if name == target_variable else ""
        lines.append(f"- `{name}`: {var.dtype}, shape={var.shape}{marker}")

    # Dimensions
    lines.append("\n## Detected Dimensions")
    lines.append(f"- Time: `{dimensions['time']}` (size={ds.dims[dimensions['time']]})")
    lines.append(f"- Latitude: `{dimensions['lat']}` (size={ds.dims[dimensions['lat']]})")
    lines.append(f"- Longitude: `{dimensions['lon']}` (size={ds.dims[dimensions['lon']]})")

    # Time coverage
    time_coord = ds.coords[dimensions["time"]]
    lines.append("\n## Time Coverage")
    lines.append(f"- Start: {time_coord.values[0]}")
    lines.append(f"- End: {time_coord.values[-1]}")
    lines.append(f"- Steps: {len(time_coord)}")

    # Spatial resolution
    lat_coord = ds.coords[dimensions["lat"]]
    lon_coord = ds.coords[dimensions["lon"]]
    if len(lat_coord) > 1:
        lat_res = abs(float(lat_coord.values[1] - lat_coord.values[0]))
        lon_res = abs(float(lon_coord.values[1] - lon_coord.values[0]))
        lines.append(f"\n## Spatial Resolution")
        lines.append(f"- Lat resolution: {lat_res:.4f}°")
        lines.append(f"- Lon resolution: {lon_res:.4f}°")
        lines.append(f"- Lat range: [{float(lat_coord.min()):.2f}, {float(lat_coord.max()):.2f}]")
        lines.append(f"- Lon range: [{float(lon_coord.min()):.2f}, {float(lon_coord.max()):.2f}]")

    # Missing value statistics
    lines.append("\n## Missing Value Statistics")
    lines.append(f"- Total pixels: {mask_metadata['total_pixels']:,}")
    lines.append(f"- Missing pixels: {mask_metadata['total_missing']:,}")
    lines.append(f"- Observation fraction: {mask_metadata['observation_fraction'] * 100:.1f}%")
    lines.append(f"- Missing fraction: {mask_metadata['missing_fraction'] * 100:.1f}%")
    if mask_metadata["detection_methods"]:
        lines.append(f"- Detection methods:")
        for method in mask_metadata["detection_methods"]:
            lines.append(f"  - {method}")

    # Value statistics (for observed pixels only)
    valid_values = da.values[~np.isnan(da.values.astype(np.float64))]
    if len(valid_values) > 0:
        lines.append("\n## Value Statistics (Observed Pixels)")
        lines.append(f"- Min: {float(np.nanmin(valid_values)):.4f}")
        lines.append(f"- Max: {float(np.nanmax(valid_values)):.4f}")
        lines.append(f"- Mean: {float(np.nanmean(valid_values)):.4f}")
        lines.append(f"- Std: {float(np.nanstd(valid_values)):.4f}")
        lines.append(f"- Median: {float(np.nanmedian(valid_values)):.4f}")

    report = "\n".join(lines)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info("Dataset report saved to %s", save_path)

    return report
