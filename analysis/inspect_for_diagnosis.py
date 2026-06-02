"""
Phase 1 Diagnostic Inspection Script.
Inspects the NetCDF dataset for:
  - Time variable existence and format
  - observed_mask shape
  - LR/HR value distributions
  - Ocean mask statistics
  - Data quality metrics
  - Spectral characteristics of LR vs HR
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import netCDF4 as nc

NC_PATH = r"E:\SR-GAN\IR_wind_23_24_new_SRGAN_ready.nc"

def main():
    ds = nc.Dataset(NC_PATH, "r")
    report = {}

    # 1. List all variables and dimensions
    print("=" * 60)
    print("DATASET STRUCTURE")
    print("=" * 60)
    print(f"  Dimensions: {dict(ds.dimensions)}")
    for dim_name, dim in ds.dimensions.items():
        print(f"    {dim_name}: size={len(dim)}, unlimited={dim.isunlimited()}")
    
    print(f"\n  Variables ({len(ds.variables)}):")
    var_info = {}
    for name, var in ds.variables.items():
        info = {
            "shape": list(var.shape),
            "dtype": str(var.dtype),
            "dimensions": list(var.dimensions),
        }
        # Check for attributes
        attrs = {}
        for attr in var.ncattrs():
            val = var.getncattr(attr)
            attrs[attr] = str(val) if not isinstance(val, (int, float)) else val
        info["attributes"] = attrs
        var_info[name] = info
        print(f"    {name}: shape={var.shape}, dtype={var.dtype}, dims={var.dimensions}")
        if attrs:
            for k, v in attrs.items():
                print(f"      {k}: {v}")
    report["variables"] = var_info

    # 2. Time variable investigation
    print("\n" + "=" * 60)
    print("TIME VARIABLE INVESTIGATION")
    print("=" * 60)
    time_candidates = ["time", "Time", "TIME", "date", "Date", "timestamp"]
    found_time = None
    for tc in time_candidates:
        if tc in ds.variables:
            found_time = tc
            break
    
    # Also check all variables for time-like names
    for name in ds.variables:
        if "time" in name.lower() or "date" in name.lower():
            found_time = name
    
    if found_time:
        tvar = ds.variables[found_time]
        print(f"  Found time variable: '{found_time}'")
        print(f"    Shape: {tvar.shape}")
        print(f"    Dtype: {tvar.dtype}")
        print(f"    First 5 values: {tvar[:5]}")
        print(f"    Last 5 values: {tvar[-5:]}")
        if hasattr(tvar, 'units'):
            print(f"    Units: {tvar.units}")
        if hasattr(tvar, 'calendar'):
            print(f"    Calendar: {tvar.calendar}")
        report["time_variable"] = found_time
    else:
        print("  NO TIME VARIABLE FOUND")
        print("  Checked: time, Time, TIME, date, Date, timestamp")
        print("  Also checked all variable names for 'time' or 'date' substrings")
        report["time_variable"] = None

    # 3. observed_mask investigation
    print("\n" + "=" * 60)
    print("OBSERVED_MASK INVESTIGATION")
    print("=" * 60)
    if "observed_mask" in ds.variables:
        obs = ds.variables["observed_mask"]
        print(f"  Shape: {obs.shape}")
        print(f"  Dtype: {obs.dtype}")
        print(f"  Dimensions: {obs.dimensions}")
        
        # Load a sample to check values
        if len(obs.shape) == 3:
            sample = obs[0]
            print(f"  Per-timestep: YES (first timestep shape: {sample.shape})")
        elif len(obs.shape) == 2:
            sample = obs[:]
            print(f"  Static (2D): YES")
        
        if isinstance(sample, np.ma.MaskedArray):
            sample = sample.filled(0)
        unique_vals = np.unique(sample)
        print(f"  Unique values: {unique_vals[:20]}")
        print(f"  Mean: {sample.mean():.4f}")
        print(f"  Fraction observed: {(sample > 0.5).mean():.4f}")
        report["observed_mask_shape"] = list(obs.shape)
    else:
        print("  NOT FOUND")
        # Check for similar names
        for name in ds.variables:
            if "observed" in name.lower() or "obs" in name.lower():
                print(f"  Similar variable found: {name}")
        report["observed_mask_shape"] = None

    # 4. LR/HR value distribution analysis
    print("\n" + "=" * 60)
    print("VALUE DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    lr_data = ds.variables["wind_speed_lr_norm"]
    hr_data = ds.variables["wind_speed_hr_norm"]
    mask_data = ds.variables["hr_ocean_mask"][:]
    if isinstance(mask_data, np.ma.MaskedArray):
        mask_data = mask_data.filled(0)
    
    # Sample 10 timesteps for analysis
    sample_indices = [0, 39, 78, 117, 156, 195, 234, 273, 312, 351]
    
    lr_stats = {"min": [], "max": [], "mean": [], "std": [], "fill_fraction": []}
    hr_stats = {"min": [], "max": [], "mean": [], "std": [], "fill_fraction": []}
    
    for t in sample_indices:
        lr_t = lr_data[t]
        hr_t = hr_data[t]
        if isinstance(lr_t, np.ma.MaskedArray):
            lr_t = lr_t.filled(-9999)
        if isinstance(hr_t, np.ma.MaskedArray):
            hr_t = hr_t.filled(-9999)
        
        lr_valid = lr_t[lr_t > -9990]
        hr_valid = hr_t[hr_t > -9990]
        
        lr_stats["min"].append(float(lr_valid.min()) if len(lr_valid) > 0 else np.nan)
        lr_stats["max"].append(float(lr_valid.max()) if len(lr_valid) > 0 else np.nan)
        lr_stats["mean"].append(float(lr_valid.mean()) if len(lr_valid) > 0 else np.nan)
        lr_stats["std"].append(float(lr_valid.std()) if len(lr_valid) > 0 else np.nan)
        lr_stats["fill_fraction"].append(float((lr_t <= -9990).sum()) / lr_t.size)
        
        hr_valid_ocean = hr_t.flatten()[mask_data.flatten() > 0.5]
        hr_valid_ocean = hr_valid_ocean[hr_valid_ocean > -9990]
        hr_stats["min"].append(float(hr_valid_ocean.min()) if len(hr_valid_ocean) > 0 else np.nan)
        hr_stats["max"].append(float(hr_valid_ocean.max()) if len(hr_valid_ocean) > 0 else np.nan)
        hr_stats["mean"].append(float(hr_valid_ocean.mean()) if len(hr_valid_ocean) > 0 else np.nan)
        hr_stats["std"].append(float(hr_valid_ocean.std()) if len(hr_valid_ocean) > 0 else np.nan)
        hr_stats["fill_fraction"].append(float((hr_t <= -9990).sum()) / hr_t.size)
    
    print(f"  LR (sampled {len(sample_indices)} timesteps):")
    print(f"    Value range: [{np.nanmin(lr_stats['min']):.4f}, {np.nanmax(lr_stats['max']):.4f}]")
    print(f"    Mean of means: {np.nanmean(lr_stats['mean']):.4f}")
    print(f"    Mean of stds:  {np.nanmean(lr_stats['std']):.4f}")
    print(f"    Fill fraction range: [{min(lr_stats['fill_fraction']):.4f}, {max(lr_stats['fill_fraction']):.4f}]")
    
    print(f"\n  HR (sampled {len(sample_indices)} timesteps, ocean only):")
    print(f"    Value range: [{np.nanmin(hr_stats['min']):.4f}, {np.nanmax(hr_stats['max']):.4f}]")
    print(f"    Mean of means: {np.nanmean(hr_stats['mean']):.4f}")
    print(f"    Mean of stds:  {np.nanmean(hr_stats['std']):.4f}")
    print(f"    Fill fraction range: [{min(hr_stats['fill_fraction']):.4f}, {max(hr_stats['fill_fraction']):.4f}]")
    
    report["lr_stats"] = {k: [float(x) for x in v] for k, v in lr_stats.items()}
    report["hr_stats"] = {k: [float(x) for x in v] for k, v in hr_stats.items()}

    # 5. Ocean mask statistics
    print("\n" + "=" * 60)
    print("OCEAN MASK STATISTICS")
    print("=" * 60)
    print(f"  Shape: {mask_data.shape}")
    print(f"  Ocean fraction: {(mask_data > 0.5).mean():.4f}")
    print(f"  Land fraction:  {(mask_data <= 0.5).mean():.4f}")
    print(f"  Ocean pixels:   {(mask_data > 0.5).sum()}")
    print(f"  Land pixels:    {(mask_data <= 0.5).sum()}")
    report["ocean_fraction"] = float((mask_data > 0.5).mean())

    # 6. LR ocean fraction analysis
    print("\n" + "=" * 60)
    print("LR OCEAN FRACTION ANALYSIS")
    print("=" * 60)
    lr_frac = ds.variables["lr_ocean_fraction"][:]
    if isinstance(lr_frac, np.ma.MaskedArray):
        lr_frac = lr_frac.filled(0)
    print(f"  Shape: {lr_frac.shape}")
    print(f"  Min: {lr_frac.min():.4f}")
    print(f"  Max: {lr_frac.max():.4f}")
    print(f"  Mean: {lr_frac.mean():.4f}")
    print(f"  Fraction >= 0.5: {(lr_frac >= 0.5).mean():.4f}")
    print(f"  Fraction == 1.0: {(lr_frac >= 0.99).mean():.4f}")
    print(f"  Fraction == 0.0: {(lr_frac < 0.01).mean():.4f}")

    # 7. Spectral analysis: compare LR and HR for one timestep
    print("\n" + "=" * 60)
    print("SPECTRAL ANALYSIS (Timestep 0)")
    print("=" * 60)
    hr_t0 = hr_data[0]
    lr_t0 = lr_data[0]
    if isinstance(hr_t0, np.ma.MaskedArray):
        hr_t0 = hr_t0.filled(0)
    if isinstance(lr_t0, np.ma.MaskedArray):
        lr_t0 = lr_t0.filled(0)
    
    # Replace fill values
    hr_t0 = np.where(hr_t0 <= -9990, 0, hr_t0)
    lr_t0 = np.where(lr_t0 <= -9990, 0, lr_t0)
    
    # 2D FFT power spectrum
    hr_fft = np.fft.fft2(hr_t0)
    hr_power = np.abs(hr_fft) ** 2
    hr_power_log = np.log10(hr_power + 1e-10)
    
    lr_fft = np.fft.fft2(lr_t0)
    lr_power = np.abs(lr_fft) ** 2
    lr_power_log = np.log10(lr_power + 1e-10)
    
    print(f"  HR FFT power: mean={hr_power_log.mean():.2f}, max={hr_power_log.max():.2f}")
    print(f"  LR FFT power: mean={lr_power_log.mean():.2f}, max={lr_power_log.max():.2f}")
    
    # Radially averaged power spectrum for HR
    h, w = hr_t0.shape
    cy, cx = h // 2, w // 2
    hr_fft_shift = np.fft.fftshift(hr_fft)
    hr_power_shift = np.abs(hr_fft_shift) ** 2
    
    y, x = np.mgrid[:h, :w]
    r = np.sqrt((y - cy)**2 + (x - cx)**2).astype(int)
    max_r = min(cy, cx)
    
    radial_profile = np.zeros(max_r)
    for ri in range(max_r):
        ring = hr_power_shift[r == ri]
        if len(ring) > 0:
            radial_profile[ri] = ring.mean()
    
    # Find where power drops significantly (information loss indicator)
    power_db = 10 * np.log10(radial_profile + 1e-10)
    peak_power = power_db[1:10].max()
    noise_floor = power_db[-10:].mean()
    dynamic_range = peak_power - noise_floor
    
    print(f"  HR Radial spectrum: peak={peak_power:.1f} dB, floor={noise_floor:.1f} dB, range={dynamic_range:.1f} dB")

    # 8. Global attributes
    print("\n" + "=" * 60)
    print("GLOBAL ATTRIBUTES")
    print("=" * 60)
    for attr in ds.ncattrs():
        print(f"  {attr}: {ds.getncattr(attr)}")

    ds.close()

    # Save report JSON
    out_path = os.path.join(os.path.dirname(__file__), "diagnosis_data.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Diagnostic data saved: {out_path}")


if __name__ == "__main__":
    main()
