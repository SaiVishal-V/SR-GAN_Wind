"""
Generate NetCDF outputs and sample day images from a trained WindGapGAN model.
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
from datasets.variable_discovery import discover_dimensions, detect_mask_variable, detect_missing_values
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
    
    # 1. Load Data
    logger.info(f"Loading data from {config.data.nc_path}")
    ds = load_dataset(config.data.nc_path)
    target_var = config.data.target_variable
    da = ds[target_var]
    
    mask_var = config.data.mask_variable or detect_mask_variable(ds)
    if mask_var and mask_var in ds.data_vars:
        real_mask = ds[mask_var].values.astype(np.float32)
        real_mask = (real_mask > 0).astype(np.float32)
    else:
        real_mask, _ = detect_missing_values(da, user_sentinels=config.data.missing_values)
        
    data = da.values.astype(np.float32)
    T, H, W = data.shape
    
    if real_mask.ndim == 2:
        real_mask = np.broadcast_to(real_mask, (T, H, W)).astype(np.float32)
        
    # Temporal splitting to get test set
    train_end = int(T * config.data.train_ratio)
    val_end = int(T * (config.data.train_ratio + config.data.val_ratio))
    
    test_data = data[val_end:]
    test_mask = real_mask[val_end:]
    
    dims = discover_dimensions(ds)
    test_time_coords = ds.coords[dims["time"]].values[val_end:]
    lat_coords = ds.coords[dims["lat"]].values
    lon_coords = ds.coords[dims["lon"]].values
    
    # Recompute norm stats from training set
    train_data = data[:train_end]
    train_mask = real_mask[:train_end]
    train_valid = train_data[train_mask > 0]
    from datasets.nc_dataset import compute_normalization_stats, normalize
    norm_stats = compute_normalization_stats(train_valid, method=config.data.norm_method)
    
    # Normalize test data
    test_data_norm = normalize(test_data, norm_stats)
    test_data_norm = np.nan_to_num(test_data_norm, nan=0.0)
    
    # 2. Load Model
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
    
    # 3. Generate Predictions on Full Images (Fully Convolutional)
    logger.info("Generating predictions on test set (full spatial grid)...")
    predictions = np.zeros_like(test_data)
    
    output_dir = Path(config.output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "sample_images"
    images_dir.mkdir(exist_ok=True)
    
    with torch.no_grad():
        for i in range(len(test_data)):
            # Shape: (1, 2, H, W) -> [field, mask]
            t_input = test_data_norm[i] * test_mask[i]
            t_mask = test_mask[i]
            
            x = torch.from_numpy(t_input).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
            m = torch.from_numpy(t_mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            model_input = torch.cat([x, m], dim=1).to(device)
            
            # Predict
            pred_norm = model(model_input).cpu().numpy()[0, 0] # (H, W)
            
            # Denormalize
            pred = denormalize(pred_norm, norm_stats)
            predictions[i] = pred
            
            # Save a few sample images (e.g. first 5 days of test set)
            if i < 5:
                fig = plot_prediction_comparison(
                    ground_truth=test_data[i],
                    prediction=pred,
                    mask=test_mask[i],
                    title=f"Test Set Day {i+1} Gap Filling",
                )
                fig.savefig(images_dir / f"reconstruction_day_{i+1}.png", bbox_inches='tight')
                plt.close(fig)
                
    logger.info(f"Sample images saved to {images_dir}")
    
    # 4. Save to NetCDF
    nc_out_path = output_dir / "test_predictions.nc"
    logger.info(f"Saving NetCDF to {nc_out_path}...")
    save_predictions_netcdf(
        save_path=nc_out_path,
        predictions=predictions,
        ground_truth=test_data,
        observed_mask=test_mask,
        time_coords=test_time_coords,
        lat_coords=lat_coords,
        lon_coords=lon_coords,
        variable_name=target_var,
    )
    logger.info("Done!")

if __name__ == "__main__":
    main()
