"""
Generate NetCDF outputs and sample day images from a trained WindGapGAN model.

This script:
1. Loads the full dataset and builds proper observation + land masks
2. Runs the model on each test timestep (full spatial grid)
3. Applies land masking (land pixels → NaN in output)
4. Saves predictions to NetCDF
5. Generates sample day comparison images
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.config import parse_args_and_load_config
from utils.logging_utils import setup_logging
from models.unet import MaskedUNet
from utils.checkpoint import CheckpointManager
from datasets.nc_dataset import denormalize
from utils.io import save_predictions_netcdf, load_dataset
from datasets.variable_discovery import (
    discover_dimensions, detect_mask_variable, detect_missing_values,
    discover_variables, prompt_variable_selection,
)
from visualization.prediction_maps import plot_prediction_comparison

def main():
    parser = argparse.ArgumentParser(description="Generate NC outputs and images")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    known_args, remaining = parser.parse_known_args()
    
    sys.argv = [sys.argv[0]] + remaining
    config = parse_args_and_load_config()
    setup_logging(log_dir=config.logging.log_dir)
    logger = logging.getLogger(__name__)
    
    device = config.resolve_device()
    
    # ── 1. Load Data ──────────────────────────────────────────────────
    logger.info(f"Loading data from {config.data.nc_path}")
    ds = load_dataset(config.data.nc_path)
    
    variables = discover_variables(ds)
    target_var = prompt_variable_selection(variables, auto_select=config.data.target_variable)
    da = ds[target_var]
    data = da.values.astype(np.float32)
    T, H, W = data.shape

    # ── Build Land Mask + Observation Mask ────────────────────────────
    mask_var = config.data.mask_variable or detect_mask_variable(ds)
    land_mask = None
    
    if mask_var and mask_var in ds.data_vars:
        raw_mask = ds[mask_var].values.astype(np.float32)
        raw_mask = (raw_mask > 0).astype(np.float32)
        
        if raw_mask.ndim == 2:
            land_mask = raw_mask.copy()  # (H, W), 1=ocean, 0=land
            nan_mask = (~np.isnan(data)).astype(np.float32)
            obs_mask = land_mask[np.newaxis, :, :] * nan_mask
            logger.info("Land mask: %.1f%% ocean, Obs mask: %.1f%% observed",
                        100 * land_mask.mean(), 100 * obs_mask.mean())
        else:
            obs_mask = raw_mask
    else:
        obs_mask, _ = detect_missing_values(da, user_sentinels=config.data.missing_values)
        
    # ── Temporal splitting ────────────────────────────────────────────
    train_end = int(T * config.data.train_ratio)
    val_end = int(T * (config.data.train_ratio + config.data.val_ratio))
    
    test_data = data[val_end:]
    test_mask = obs_mask[val_end:]
    
    dims = discover_dimensions(ds)
    test_time_coords = ds.coords[dims["time"]].values[val_end:]
    lat_coords = ds.coords[dims["lat"]].values
    lon_coords = ds.coords[dims["lon"]].values
    
    # ── Normalization (from train set only) ───────────────────────────
    train_data = data[:train_end]
    train_mask_data = obs_mask[:train_end]
    train_valid = train_data[train_mask_data > 0]
    from datasets.nc_dataset import compute_normalization_stats, normalize
    norm_stats = compute_normalization_stats(train_valid, method=config.data.norm_method)
    
    test_data_norm = normalize(np.nan_to_num(test_data, nan=0.0), norm_stats)
    
    # ── 2. Load Model ────────────────────────────────────────────────
    model = MaskedUNet(
        in_channels=config.model.in_channels,
        out_channels=config.model.out_channels,
        base_features=config.model.base_features,
        depth=config.model.depth,
        dropout=0.0,
        use_batch_norm=config.model.use_batch_norm,
    ).to(device)
    
    CheckpointManager.load(known_args.checkpoint, model, device=device)
    model.eval()
    logger.info("Model loaded successfully.")
    
    # ── 3. Generate Predictions ───────────────────────────────────────
    logger.info("Generating predictions on test set (%d timesteps)...", len(test_data))
    predictions = np.zeros_like(test_data)
    
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "sample_images"
    images_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for i in range(len(test_data)):
            # Build input: (1, 2, H, W) — [masked_field, obs_mask]
            field = test_data_norm[i] * test_mask[i]   # Zero out gaps
            mask = test_mask[i]                         # Observation mask
            
            x = torch.from_numpy(field).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            m = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)   # (1, 1, H, W)
            model_input = torch.cat([x, m], dim=1).to(device)      # (1, 2, H, W)
            
            pred_norm = model(model_input).cpu().numpy()[0, 0]  # (H, W)
            pred = denormalize(pred_norm, norm_stats)
            
            # Apply land mask: set land pixels to NaN
            if land_mask is not None:
                pred = np.where(land_mask > 0, pred, np.nan)
            
            predictions[i] = pred
            
            # Save sample images (first 5 test days)
            if i < 5:
                gt_display = np.where(land_mask > 0, test_data[i], np.nan) if land_mask is not None else test_data[i]
                fig = plot_prediction_comparison(
                    ground_truth=gt_display,
                    prediction=pred,
                    mask=test_mask[i],
                    title=f"Test Day {i+1} — Gap Filling",
                )
                fig.savefig(images_dir / f"reconstruction_day_{i+1}.png",
                            dpi=150, bbox_inches='tight')
                plt.close(fig)
                
    logger.info(f"Sample images saved to {images_dir}")
    
    # ── 4. Save to NetCDF ─────────────────────────────────────────────
    nc_out_path = output_dir / "test_predictions.nc"
    logger.info(f"Saving NetCDF to {nc_out_path}...")
    
    # Apply land mask to ground truth too for consistency
    gt_out = test_data.copy()
    if land_mask is not None:
        gt_out = np.where(land_mask[np.newaxis, :, :] > 0, gt_out, np.nan)
    
    save_predictions_netcdf(
        save_path=nc_out_path,
        predictions=predictions,
        ground_truth=gt_out,
        observed_mask=test_mask,
        time_coords=test_time_coords,
        lat_coords=lat_coords,
        lon_coords=lon_coords,
        variable_name=target_var,
    )
    logger.info("Done! Outputs saved to %s", output_dir)

if __name__ == "__main__":
    main()
