# WindGapGAN — Spatio-Temporal Gap Filling Framework

> **Dataset-agnostic Earth Observation gap filling** using GANFilling-inspired architectures.
>
> Adapted for ocean surface wind speed, generalizable to any NetCDF variable.

## Scientific Objective

```
Input:  Wind field with missing observations + Observation mask
Output: Gap-filled wind field (observed pixels preserved exactly)
```

This is a **sequence-to-sequence gap-filling** problem, not super-resolution.

## Reference

> Gonzalez-Calabuig, M., Fernández-Torres, M.-Á., & Camps-Valls, G. (2025).
> *Generative Networks for Spatio-Temporal Gap Filling of Sentinel-2 Reflectances*.
> ISPRS Journal of Photogrammetry and Remote Sensing, 220, 637–648.

## Architecture Phases

| Phase | Model | Purpose |
|-------|-------|---------|
| 1 | Masked U-Net | Spatial baseline (no temporal modeling) |
| 2 | ConvLSTM U-Net | Spatio-temporal modeling |
| 3 | ConvLSTM U-Net + PatchGAN | Adversarial gap filling |

## Quick Start

### Installation

```bash
# Option A: pip
pip install -r requirements.txt

# Option B: conda
conda env create -f environment.yml
conda activate windgapgan
```

### Explore Your Dataset

```bash
python scripts/inspect_dataset.py --nc_path /path/to/your/data.nc
```

### Train

```bash
# Phase 1: Masked U-Net baseline
python train.py --config configs/default.yaml --data.nc_path /path/to/data.nc

# Phase 2: ConvLSTM U-Net
python train.py --config configs/default.yaml --model.name convlstm_unet --data.nc_path /path/to/data.nc

# Phase 3: GAN
python train.py --config configs/default.yaml --model.name gan --data.nc_path /path/to/data.nc
```

### Evaluate

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_rmse.pt
```

## Features

- **Dataset Agnostic**: Works with any NetCDF file — variables, dimensions, and missing values discovered automatically
- **Adaptive Normalization**: `auto | zscore | minmax | robust | none`
- **Automatic Mask Detection**: Priority chain: explicit mask → `_FillValue` → `NaN` / sentinel values
- **Classical Baselines**: Persistence, Linear Interpolation, Nearest Neighbor, Mean Filling
- **Gap-Only Evaluation**: Separate metrics for reconstructed vs. observed regions
- **Error Stratification**: Metrics broken down by wind-speed regime
- **Distribution Metrics**: KL Divergence, Wasserstein Distance
- **Experiment Tracking**: TensorBoard + Weights & Biases

## Project Structure

```
WindGapGAN/
├── configs/         # YAML configuration files
├── datasets/        # NetCDF dataset, variable discovery, mask generation
├── models/          # U-Net, ConvLSTM U-Net, Generator, Discriminator
├── losses/          # Masked L1, Gradient, Adversarial losses
├── trainers/        # Training loops (baseline, ConvLSTM, GAN)
├── evaluators/      # Metrics and evaluation pipeline
├── visualization/   # Loss plots, prediction maps, error maps
├── utils/           # Config, logging, checkpointing, I/O
├── scripts/         # CLI tools
├── tests/           # Unit tests
├── train.py         # Main training entry point
└── evaluate.py      # Main evaluation entry point
```

## License

MIT
