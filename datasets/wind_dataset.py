"""
WindSRDataset — NetCDF-backed PyTorch Dataset for wind-speed super-resolution (V2).

Features:
    - Efficient NetCDF reading via netCDF4 with memory mapping
    - Optional full-dataset caching in RAM
    - Patch extraction with configurable ocean fraction threshold
    - Temporal train/val/test splitting (no random split — prevents leakage)
    - Full-scene mode for validation/evaluation
    - Proper handling of _FillValue (-9999) and masked arrays
    - observed_mask loading for observed-only evaluation
    - No data augmentation (geophysical constraint)

Verified variable names and shapes from dataset inspection:
    - wind_speed_lr_norm: (392, 80, 140), _FillValue=-9999.0
    - wind_speed_hr_norm: (392, 320, 560), _FillValue=-9999.0
    - hr_ocean_mask: (320, 560), binary {0,1}, 1=ocean
    - lr_ocean_fraction: (80, 140), quantized 1/16 steps
    - observed_mask: (392, 321, 561), binary {0,1}, per-timestep
"""

import numpy as np
import netCDF4 as nc
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class WindSRDataset(Dataset):
    """
    PyTorch Dataset for wind-speed super-resolution.

    Supports two modes:
        - patch: Extracts random patches per timestep (for training)
        - full: Returns full LR/HR scenes (for validation/evaluation)
    """

    def __init__(
        self,
        nc_path: str,
        lr_variable: str = "wind_speed_lr_norm",
        hr_variable: str = "wind_speed_hr_norm",
        mask_variable: str = "hr_ocean_mask",
        quality_variable: str = "lr_ocean_fraction",
        time_indices: Optional[List[int]] = None,
        mode: str = "patch",
        lr_patch_size: int = 16,
        hr_patch_size: int = 64,
        min_ocean_fraction: float = 0.5,
        patches_per_image: int = 32,
        scale_factor: int = 4,
        cache: bool = True,
        load_observed_mask: bool = False,
    ):
        """
        Args:
            nc_path: Path to the NetCDF dataset.
            lr_variable: Name of the LR wind speed variable.
            hr_variable: Name of the HR wind speed variable.
            mask_variable: Name of the HR ocean mask variable.
            quality_variable: Name of the LR ocean fraction variable.
            time_indices: List of time indices to use (for train/val/test split).
            mode: 'patch' for patch extraction, 'full' for full scenes.
            lr_patch_size: Spatial size of LR patches.
            hr_patch_size: Spatial size of HR patches.
            min_ocean_fraction: Minimum ocean fraction for patch acceptance.
            patches_per_image: Number of random patches per timestep per epoch.
            scale_factor: Super-resolution scale factor.
            cache: If True, cache entire dataset in RAM.
            load_observed_mask: If True, load per-timestep observed_mask for
                observed-only evaluation.
        """
        super().__init__()

        self.nc_path = nc_path
        self.mode = mode
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = hr_patch_size
        self.min_ocean_fraction = min_ocean_fraction
        self.patches_per_image = patches_per_image
        self.scale_factor = scale_factor
        self.load_observed_mask = load_observed_mask

        # Open dataset and read metadata
        ds = nc.Dataset(nc_path, "r")

        # Determine time indices
        n_time = ds.variables[lr_variable].shape[0]
        if time_indices is not None:
            self.time_indices = time_indices
        else:
            self.time_indices = list(range(n_time))

        # Read ocean mask (static, 2D) — always loaded
        mask_data = ds.variables[mask_variable][:]
        if isinstance(mask_data, np.ma.MaskedArray):
            mask_data = mask_data.filled(0)
        self.hr_ocean_mask = mask_data.astype(np.float32)

        # Read LR ocean fraction (static, 2D)
        quality_data = ds.variables[quality_variable][:]
        if isinstance(quality_data, np.ma.MaskedArray):
            quality_data = quality_data.filled(0)
        self.lr_ocean_fraction = quality_data.astype(np.float32)

        # Load observed_mask if requested
        # Shape: (392, 321, 561) on original grid — must crop to (320, 560)
        self.observed_mask_cache = None
        if load_observed_mask and "observed_mask" in ds.variables:
            obs_var = ds.variables["observed_mask"]
            obs_shape = obs_var.shape

            if len(obs_shape) == 3:
                # Per-timestep: (time, lat_orig, lon_orig)
                # Need to crop from (321, 561) to (320, 560) to match HR grid
                hr_h, hr_w = self.hr_ocean_mask.shape
                obs_data = obs_var[self.time_indices, :hr_h, :hr_w]
                if isinstance(obs_data, np.ma.MaskedArray):
                    obs_data = obs_data.filled(0)
                self.observed_mask_cache = obs_data.astype(np.float32)
            elif len(obs_shape) == 2:
                # Static: (lat, lon) — broadcast across timesteps
                obs_data = obs_var[:self.hr_ocean_mask.shape[0], :self.hr_ocean_mask.shape[1]]
                if isinstance(obs_data, np.ma.MaskedArray):
                    obs_data = obs_data.filled(0)
                self.observed_mask_cache = obs_data.astype(np.float32)

        # Load time variable for monthly evaluation
        self.time_values = None
        if "time" in ds.variables:
            time_var = ds.variables["time"]
            self.time_values = time_var[self.time_indices]
            self.time_units = getattr(time_var, "units", None)
            self.time_calendar = getattr(time_var, "calendar", "gregorian")

        # Precompute valid patch positions (LR coordinates where ocean_fraction >= threshold)
        if mode == "patch":
            self._precompute_valid_positions()

        # Cache or lazy load
        self.cache = cache
        if cache:
            lr_data = ds.variables[lr_variable][self.time_indices]
            hr_data = ds.variables[hr_variable][self.time_indices]

            if isinstance(lr_data, np.ma.MaskedArray):
                lr_data = lr_data.filled(0.0)
            if isinstance(hr_data, np.ma.MaskedArray):
                hr_data = hr_data.filled(0.0)

            self.lr_cache = lr_data.astype(np.float32)
            self.hr_cache = hr_data.astype(np.float32)
            ds.close()
            self._ds = None
        else:
            self._ds = ds
            self._lr_var = lr_variable
            self._hr_var = hr_variable
            self.lr_cache = None
            self.hr_cache = None

    def _precompute_valid_positions(self) -> None:
        """Find all LR grid positions where ocean fraction >= threshold."""
        h_lr, w_lr = self.lr_ocean_fraction.shape
        self.valid_positions = []

        for y in range(h_lr - self.lr_patch_size + 1):
            for x in range(w_lr - self.lr_patch_size + 1):
                patch_frac = self.lr_ocean_fraction[
                    y : y + self.lr_patch_size, x : x + self.lr_patch_size
                ]
                mean_frac = patch_frac.mean()
                if mean_frac >= self.min_ocean_fraction:
                    self.valid_positions.append((y, x))

        if len(self.valid_positions) == 0:
            raise ValueError(
                f"No valid patches found with min_ocean_fraction={self.min_ocean_fraction}. "
                f"Try lowering the threshold."
            )

    def _load_timestep(self, local_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load LR and HR arrays for a given local index."""
        if self.cache:
            lr = self.lr_cache[local_idx]
            hr = self.hr_cache[local_idx]
        else:
            t = self.time_indices[local_idx]
            lr = self._ds.variables[self._lr_var][t]
            hr = self._ds.variables[self._hr_var][t]

            if isinstance(lr, np.ma.MaskedArray):
                lr = lr.filled(0.0)
            if isinstance(hr, np.ma.MaskedArray):
                hr = hr.filled(0.0)

            lr = lr.astype(np.float32)
            hr = hr.astype(np.float32)

        return lr, hr

    def __len__(self) -> int:
        if self.mode == "patch":
            return len(self.time_indices) * self.patches_per_image
        else:
            return len(self.time_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.mode == "patch":
            return self._get_patch(idx)
        else:
            return self._get_full(idx)

    def _get_patch(self, idx: int) -> Dict[str, torch.Tensor]:
        """Extract a random valid patch from a timestep."""
        t_idx = idx // self.patches_per_image
        lr, hr = self._load_timestep(t_idx)

        # Random valid position
        pos_idx = np.random.randint(len(self.valid_positions))
        y_lr, x_lr = self.valid_positions[pos_idx]

        # Extract LR patch
        lr_patch = lr[
            y_lr : y_lr + self.lr_patch_size,
            x_lr : x_lr + self.lr_patch_size,
        ]

        # Corresponding HR patch
        y_hr = y_lr * self.scale_factor
        x_hr = x_lr * self.scale_factor
        hr_patch = hr[
            y_hr : y_hr + self.hr_patch_size,
            x_hr : x_hr + self.hr_patch_size,
        ]

        # Corresponding mask patch
        mask_patch = self.hr_ocean_mask[
            y_hr : y_hr + self.hr_patch_size,
            x_hr : x_hr + self.hr_patch_size,
        ]

        # Replace any remaining fill values with 0
        lr_patch = np.where(lr_patch <= -9990, 0.0, lr_patch)
        hr_patch = np.where(hr_patch <= -9990, 0.0, hr_patch)

        # Convert to tensors with channel dimension: (1, H, W)
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_patch).unsqueeze(0)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "mask": mask_tensor,
        }

    def _get_full(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a full LR/HR scene for validation or evaluation."""
        lr, hr = self._load_timestep(idx)

        # Replace fill values
        lr = np.where(lr <= -9990, 0.0, lr)
        hr = np.where(hr <= -9990, 0.0, hr)

        lr_tensor = torch.from_numpy(lr).unsqueeze(0)
        hr_tensor = torch.from_numpy(hr).unsqueeze(0)
        mask_tensor = torch.from_numpy(self.hr_ocean_mask).unsqueeze(0)

        sample = {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "mask": mask_tensor,
        }

        # Include observed mask if loaded
        if self.observed_mask_cache is not None:
            if self.observed_mask_cache.ndim == 3:
                obs = self.observed_mask_cache[idx]
            else:
                obs = self.observed_mask_cache
            sample["observed_mask"] = torch.from_numpy(obs).unsqueeze(0)

        return sample

    def get_time_dates(self):
        """
        Convert time values to Python datetime objects.

        Returns:
            List of datetime objects, or None if time variable unavailable.

        Raises:
            RuntimeError if time variable exists but cannot be decoded.
        """
        if self.time_values is None or self.time_units is None:
            return None

        try:
            import cftime
            dates = nc.num2date(
                self.time_values,
                units=self.time_units,
                calendar=self.time_calendar,
            )
            return dates
        except Exception as e:
            raise RuntimeError(
                f"Cannot decode time variable (units='{self.time_units}', "
                f"calendar='{self.time_calendar}'): {e}\n"
                f"Time handling requires valid CF-compliant time metadata. "
                f"Please verify the dataset time variable."
            )

    def close(self) -> None:
        """Close the underlying NetCDF dataset if not cached."""
        if self._ds is not None:
            self._ds.close()
            self._ds = None


def get_temporal_split(
    n_time: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split timestep indices temporally (NOT randomly).

    Args:
        n_time: Total number of timesteps.
        train_ratio: Fraction for training (first N%).
        val_ratio: Fraction for validation (next N%).
        test_ratio: Fraction for testing (final N%).

    Returns:
        Tuple of (train_indices, val_indices, test_indices).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    n_train = int(n_time * train_ratio)
    n_val = int(n_time * val_ratio)

    train_idx = list(range(0, n_train))
    val_idx = list(range(n_train, n_train + n_val))
    test_idx = list(range(n_train + n_val, n_time))

    return train_idx, val_idx, test_idx
