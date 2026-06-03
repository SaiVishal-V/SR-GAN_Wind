"""
CLI tool to inspect any NetCDF dataset.

Usage:
    python scripts/inspect_dataset.py --nc_path /path/to/data.nc

Outputs:
    - Available variables with shapes and missing percentages
    - Detected dimensions (time, lat, lon)
    - Missing value statistics
    - Basic value statistics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.io import load_dataset, get_dataset_summary
from datasets.variable_discovery import (
    discover_variables,
    discover_dimensions,
    detect_missing_values,
    generate_dataset_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a NetCDF dataset for WindGapGAN compatibility",
    )
    parser.add_argument(
        "--nc_path",
        type=str,
        required=True,
        help="Path to the NetCDF file to inspect",
    )
    parser.add_argument(
        "--target_variable",
        type=str,
        default=None,
        help="Target variable to analyze in detail (optional)",
    )
    parser.add_argument(
        "--save_report",
        type=str,
        default=None,
        help="Path to save the dataset report (optional)",
    )
    args = parser.parse_args()

    # Load dataset
    print(f"\nLoading: {args.nc_path}")
    ds = load_dataset(args.nc_path)

    # Print basic summary
    print(get_dataset_summary(ds))

    # Discover variables
    variables = discover_variables(ds)

    # Discover dimensions
    try:
        dims = discover_dimensions(ds)
        print(f"\n✓ Detected dimensions:")
        print(f"  Time:      {dims['time']} (size={ds.dims[dims['time']]})")
        print(f"  Latitude:  {dims['lat']} (size={ds.dims[dims['lat']]})")
        print(f"  Longitude: {dims['lon']} (size={ds.dims[dims['lon']]})")
    except ValueError as e:
        print(f"\n✗ Dimension detection failed: {e}")
        dims = None

    # Analyze target variable if specified
    if args.target_variable:
        if args.target_variable not in ds.data_vars:
            print(f"\n✗ Variable '{args.target_variable}' not found.")
        else:
            da = ds[args.target_variable]
            print(f"\n{'=' * 50}")
            print(f"TARGET VARIABLE: {args.target_variable}")
            print(f"{'=' * 50}")
            print(f"  Shape: {da.shape}")
            print(f"  Dims:  {da.dims}")
            print(f"  Dtype: {da.dtype}")

            # Detect missing values
            mask, metadata = detect_missing_values(da)
            print(f"\n  Missing Value Detection:")
            print(f"    Observation fraction: {metadata['observation_fraction'] * 100:.1f}%")
            print(f"    Missing fraction:     {metadata['missing_fraction'] * 100:.1f}%")
            for method in metadata["detection_methods"]:
                print(f"    Method: {method}")

            # Generate report
            if dims:
                report = generate_dataset_report(
                    ds=ds,
                    target_variable=args.target_variable,
                    dimensions=dims,
                    mask_metadata=metadata,
                    save_path=args.save_report,
                )
                if args.save_report:
                    print(f"\n  Report saved to: {args.save_report}")

    ds.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
