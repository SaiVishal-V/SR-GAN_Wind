"""
Evaluation script for wind-speed super-resolution.

Generates:
    - Prediction vs Ground Truth scatter plot
    - Wind-speed distribution histogram
    - LR input map, SR prediction map, Ground truth map, Error map
    - Physical metrics summary table
    - Reconstructed NetCDF output (review #4: with GT, error, mask)

Usage:
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_rmse_generator.pth
    python evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_rmse_generator.pth --colab
"""

import argparse
import os
import sys
import json

import numpy as np
import torch
import yaml
import netCDF4 as nc
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.seed import set_seed
from utils.checkpoint import load_checkpoint
from utils.metrics import (
    compute_all_metrics,
    denormalize,
    NORM_MEAN,
    NORM_STD,
)
from utils.plotting import (
    plot_comparison,
    plot_scatter,
    plot_histogram,
)
from datasets.wind_dataset import WindSRDataset, get_temporal_split
from models.generator import SRResNet


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_model(
    generator: torch.nn.Module,
    test_dataset: WindSRDataset,
    device: torch.device,
    output_dir: str,
    config: dict,
) -> dict:
    """
    Full evaluation on test set.

    Generates:
        - Per-scene metrics
        - Averaged metrics
        - Comparison plots
        - Scatter and histogram plots
        - NetCDF output (review #4)
    """
    generator.eval()
    os.makedirs(output_dir, exist_ok=True)

    n_scenes = len(test_dataset)
    all_metrics = {}
    all_pred_values = []
    all_target_values = []

    # Storage for NetCDF output
    sr_all = []
    hr_all = []
    # Transpose mask to fix spatial orientation (lat/lon mapping)
    mask = test_dataset.hr_ocean_mask.T

    print(f"\nEvaluating {n_scenes} test scenes...")

    for i in tqdm(range(n_scenes), desc="Evaluating"):
        sample = test_dataset[i]
        lr = sample["lr"].unsqueeze(0).to(device)
        hr = sample["hr"].unsqueeze(0).to(device)
        mask_t = sample["mask"].unsqueeze(0).to(device)

        sr = generator(lr)

        # Compute metrics
        metrics = compute_all_metrics(sr, hr, mask_t)

        for key, val in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = []
            all_metrics[key].append(val)

        # Collect denormalized values for scatter/histogram (transpose to match mask)
        sr_np = sr.squeeze().cpu().numpy().T
        hr_np = hr.squeeze().cpu().numpy().T

        sr_mps = sr_np * NORM_STD + NORM_MEAN
        hr_mps = hr_np * NORM_STD + NORM_MEAN

        ocean = mask > 0.5
        all_pred_values.append(sr_mps[ocean])
        all_target_values.append(hr_mps[ocean])

        sr_all.append(sr_np)
        hr_all.append(hr_np)

        # Plot first 5 scenes
        if i < 5:
            lr_np = lr.squeeze().cpu().numpy().T
            plot_comparison(
                lr=lr_np,
                sr=sr_np,
                hr=hr_np,
                mask_hr=mask,
                vmin=config["evaluation"]["vmin"],
                vmax=config["evaluation"]["vmax"],
                cmap_name=config["evaluation"]["cmap"],
                save_path=os.path.join(output_dir, f"comparison_scene_{i}.png"),
                title=f"Test Scene {i}",
                is_normalized=True,
            )

    # Average metrics
    avg_metrics = {}
    print("\n" + "=" * 60)
    print("TEST SET METRICS (averaged over all scenes)")
    print("=" * 60)
    for key in sorted(all_metrics.keys()):
        avg = np.mean(all_metrics[key])
        std = np.std(all_metrics[key])
        avg_metrics[key] = {"mean": float(avg), "std": float(std)}
        print(f"  {key:20s}: {avg:.6f} ± {std:.6f}")

    # Save metrics JSON
    metrics_path = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(avg_metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")

    # Scatter plot
    all_pred = np.concatenate(all_pred_values)
    all_target = np.concatenate(all_target_values)

    plot_scatter(
        all_pred,
        all_target,
        save_path=os.path.join(output_dir, "scatter_pred_vs_truth.png"),
        title="SR-GAN: Prediction vs Ground Truth (m/s)",
    )

    # Histogram
    plot_histogram(
        all_pred,
        all_target,
        save_path=os.path.join(output_dir, "histogram_wind_speed.png"),
        title="Wind Speed Distribution (m/s)",
    )

    # --- NetCDF Output (review #4) ---
    print("\nGenerating output NetCDF...")
    save_netcdf_output(
        sr_all, hr_all, mask, output_dir, test_dataset
    )

    return avg_metrics


def save_netcdf_output(
    sr_all: list,
    hr_all: list,
    mask: np.ndarray,
    output_dir: str,
    dataset: WindSRDataset,
) -> None:
    """
    Save reconstructed predictions as NetCDF (review #4).

    Structure:
        time, lat_hr, lon_hr
        predicted_wind_speed_norm
        predicted_wind_speed_mps
        ground_truth_wind_speed_mps
        error_mps
        ocean_mask
    """
    n_time = len(sr_all)
    h, w = mask.shape

    nc_path = os.path.join(output_dir, "predictions.nc")
    ds = nc.Dataset(nc_path, "w", format="NETCDF4")

    # Dimensions
    ds.createDimension("time", n_time)
    ds.createDimension("lat_hr", h)
    ds.createDimension("lon_hr", w)

    # Variables
    v_sr_norm = ds.createVariable(
        "predicted_wind_speed_norm", "f4", ("time", "lat_hr", "lon_hr"),
        fill_value=-9999.0,
    )
    v_sr_norm.long_name = "Super-resolved wind speed (normalized)"

    v_sr_mps = ds.createVariable(
        "predicted_wind_speed_mps", "f4", ("time", "lat_hr", "lon_hr"),
        fill_value=-9999.0,
    )
    v_sr_mps.units = "m s-1"
    v_sr_mps.long_name = "Super-resolved wind speed"

    v_gt_mps = ds.createVariable(
        "ground_truth_wind_speed_mps", "f4", ("time", "lat_hr", "lon_hr"),
        fill_value=-9999.0,
    )
    v_gt_mps.units = "m s-1"
    v_gt_mps.long_name = "Ground truth wind speed"

    v_error = ds.createVariable(
        "error_mps", "f4", ("time", "lat_hr", "lon_hr"),
        fill_value=-9999.0,
    )
    v_error.units = "m s-1"
    v_error.long_name = "Prediction error (SR - GT)"

    v_mask = ds.createVariable("ocean_mask", "u1", ("lat_hr", "lon_hr"))
    v_mask.long_name = "Ocean mask (1=ocean, 0=land)"

    # Write data
    for t in range(n_time):
        sr_norm = sr_all[t]
        hr_norm = hr_all[t]

        sr_mps = sr_norm * NORM_STD + NORM_MEAN
        hr_mps = hr_norm * NORM_STD + NORM_MEAN

        # Mask land
        sr_norm_masked = np.where(mask == 1, sr_norm, -9999.0)
        sr_mps_masked = np.where(mask == 1, sr_mps, -9999.0)
        hr_mps_masked = np.where(mask == 1, hr_mps, -9999.0)
        error_masked = np.where(mask == 1, sr_mps - hr_mps, -9999.0)

        v_sr_norm[t] = sr_norm_masked
        v_sr_mps[t] = sr_mps_masked
        v_gt_mps[t] = hr_mps_masked
        v_error[t] = error_masked

    v_mask[:] = mask.astype(np.uint8)

    # Global attributes
    ds.title = "SR-GAN Wind Speed Predictions"
    ds.norm_mean = str(NORM_MEAN)
    ds.norm_std = str(NORM_STD)
    ds.denormalization_formula = "wind_speed_mps = wind_speed_norm * std + mean"

    ds.close()
    print(f"  NetCDF saved: {nc_path}")


def main():
    parser = argparse.ArgumentParser(description="SR-GAN Evaluation")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"], deterministic=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data path
    data_path = config["data"]["colab_path"] if args.colab else config["data"]["local_path"]

    # Create test dataset
    import netCDF4
    ds_nc = netCDF4.Dataset(data_path, "r")
    n_time = ds_nc.variables[config["data"]["lr_variable"]].shape[0]
    ds_nc.close()

    _, _, test_idx = get_temporal_split(
        n_time,
        config["data"]["train_ratio"],
        config["data"]["val_ratio"],
        config["data"]["test_ratio"],
    )

    test_dataset = WindSRDataset(
        nc_path=data_path,
        lr_variable=config["data"]["lr_variable"],
        hr_variable=config["data"]["hr_variable"],
        mask_variable=config["data"]["mask_variable"],
        quality_variable=config["data"]["quality_variable"],
        time_indices=test_idx,
        mode="full",
        cache=config["data"]["cache_dataset"],
    )

    # Load model
    gen_cfg = config["model"]["generator"]
    generator = SRResNet(
        in_channels=config["model"]["in_channels"],
        num_features=gen_cfg["num_features"],
        num_residual_blocks=gen_cfg["num_residual_blocks"],
        scale_factor=config["model"]["scale_factor"],
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Output directory
    if args.colab and not args.output_dir:
        output_dir = os.path.join("/content/drive/MyDrive/SR-GAN_Wind", config["evaluation"]["output_dir"])
        print(f"  Colab Mode: Routing outputs to {output_dir}")
    else:
        output_dir = args.output_dir or config["evaluation"]["output_dir"]

    # Evaluate
    evaluate_model(generator, test_dataset, device, output_dir, config)


if __name__ == "__main__":
    main()
