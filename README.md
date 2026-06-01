# SR-GAN Wind Speed Super-Resolution

Production-grade Super-Resolution GAN for ocean surface wind-speed super-resolution (4×) using satellite-derived NetCDF4 data.

## Features

- **Scientific SR-GAN**: Designed for geophysical data (NOT RGB imagery)
- **Masked Training**: All losses respect ocean/land masks
- **Two-Stage Pipeline**: Generator pretraining (L1) → GAN fine-tuning
- **GPU Optimized**: AMP mixed precision, supports T4/A100 (Google Colab)
- **Fully Reproducible**: Deterministic training with seed control
- **Modular Architecture**: Clean separation of data, models, training, evaluation
- **Comprehensive Metrics**: MAE, RMSE, PSNR, SSIM, Gradient RMSE, physical metrics (m/s)
- **Baseline Comparisons**: Automated benchmarking against Bicubic and SRResNet

## Dataset

**Input**: `IR_wind_23_24_new_SRGAN_ready.nc`

| Variable | Shape | Description |
|---|---|---|
| `wind_speed_lr_norm` | (392, 80, 140) | LR normalized input |
| `wind_speed_hr_norm` | (392, 320, 560) | HR normalized target |
| `hr_ocean_mask` | (320, 560) | Ocean mask (1=ocean, 0=land) |
| `lr_ocean_fraction` | (80, 140) | Ocean fraction per LR cell |

**Scale Factor**: 4× (80→320, 140→560)

**Normalization**: `wind_speed_mps = norm × 2.9275 + 6.3363`

## Architecture

### Generator (SRResNet)
- 8 Residual Blocks (no BatchNorm)
- PReLU activation
- 2× PixelShuffle upsampling (4× total)
- 1-channel I/O

### Discriminator (PatchGAN)
- 70×70 receptive field
- Spectral Normalization
- Patch-level realism map output

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

```bash
# Full pipeline (pretrain + GAN)
python training/train.py --config configs/config.yaml

# Google Colab
python training/train.py --config configs/config.yaml --colab

# Resume from checkpoint
python training/train.py --config configs/config.yaml --resume checkpoints/last_checkpoint.pth
```

### Evaluation

```bash
python evaluation/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_rmse_generator.pth
```

### Benchmarking

```bash
python evaluation/benchmark.py --config configs/config.yaml \
    --srresnet checkpoints/pretrain_best.pth \
    --srgan checkpoints/best_rmse_generator.pth
```

## Training Strategy

| Stage | Epochs | Loss | Optimizer |
|---|---|---|---|
| 1. Pretrain | 100 | Masked L1 | AdamW (lr=1e-4) |
| 2. GAN | 200 | L1 + Adversarial + Gradient | AdamW (lr=1e-4) |

## Loss Function

```
L_G = 1.0 × Masked_L1 + 1e-3 × Adversarial + 0.1 × Gradient_Loss
```

## Metrics

| Metric | Space | Description |
|---|---|---|
| MAE | Normalized | Mean Absolute Error |
| RMSE | Normalized | Root Mean Squared Error |
| PSNR | Normalized | Peak Signal-to-Noise Ratio |
| SSIM | Normalized | Structural Similarity |
| Gradient RMSE | Normalized | Wind front preservation |
| RMSE (m/s) | Physical | After denormalization |
| MAE (m/s) | Physical | After denormalization |
| Bias (m/s) | Physical | Systematic error |
| Correlation | Physical | Pearson correlation |

## Project Structure

```
srgan_windspeed/
├── configs/config.yaml          # All hyperparameters
├── datasets/wind_dataset.py     # NetCDF dataset & patching
├── models/
│   ├── generator.py             # SRResNet (no BatchNorm)
│   ├── discriminator.py         # PatchGAN + Spectral Norm
│   └── losses.py                # Masked losses + gradient loss
├── utils/
│   ├── metrics.py               # All evaluation metrics
│   ├── plotting.py              # matplotlib visualization
│   ├── checkpoint.py            # Save/load + best-metric tracking
│   └── seed.py                  # Reproducibility
├── training/
│   ├── train.py                 # Two-stage training pipeline
│   └── validate.py              # Patch + full-scene validation
├── evaluation/
│   ├── evaluate.py              # Full evaluation + NetCDF output
│   └── benchmark.py             # Bicubic/SRResNet/SRGAN comparison
├── requirements.txt
├── README.md
└── LICENSE
```

## Scientific Constraints

- ❌ No VGG perceptual loss
- ❌ No ImageNet normalization
- ❌ No color augmentation
- ✅ Preserve wind-speed gradients
- ✅ Preserve mesoscale structures
- ✅ Mask-aware loss computation
- ✅ Physical metric evaluation

## License

MIT
