"""
WindGapGAN — Main Training Entry Point.

Usage:
    # Phase 1: Masked U-Net baseline
    python train.py --config configs/default.yaml --data.nc_path /path/to/data.nc

    # With target variable pre-selected (non-interactive)
    python train.py --config configs/default.yaml --data.nc_path /path/to/data.nc --data.target_variable wind_speed

    # Phase 2: ConvLSTM U-Net
    python train.py --config configs/default.yaml --model.name convlstm_unet

    # Override any config parameter via CLI
    python train.py --config configs/default.yaml --training.epochs 100 --training.batch_size 4
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import parse_args_and_load_config, save_config
from utils.logging_utils import setup_logging
from utils.seed import set_seed


def main() -> None:
    """Main training entry point."""
    # ── Load Configuration ─────────────────────────────────────────
    config = parse_args_and_load_config()

    # ── Setup Logging ──────────────────────────────────────────────
    setup_logging(log_dir=config.logging.log_dir)
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("WindGapGAN — Spatio-Temporal Gap Filling")
    logger.info("=" * 60)

    # ── Validate Required Fields ───────────────────────────────────
    if config.data.nc_path is None:
        logger.error("No data path specified. Use --data.nc_path=<path>")
        sys.exit(1)

    nc_path = Path(config.data.nc_path)
    if not nc_path.exists():
        logger.error("Data file not found: %s", nc_path)
        sys.exit(1)

    # ── Set Seed ───────────────────────────────────────────────────
    set_seed(config.seed, deterministic=True)

    # ── Resolve Device ─────────────────────────────────────────────
    device = config.resolve_device()
    logger.info("Device: %s", device)

    # ── Build Datasets ─────────────────────────────────────────────
    from datasets.nc_dataset import build_datasets

    logger.info("Building datasets from: %s", nc_path)
    train_ds, val_ds, test_ds, metadata = build_datasets(
        nc_path=str(nc_path),
        target_variable=config.data.target_variable,
        sequence_length=config.data.sequence_length,
        patch_size=config.data.patch_size,
        stride=config.data.stride,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        norm_method=config.data.norm_method,
        mask_variable=config.data.mask_variable,
        missing_values=config.data.missing_values,
        synthetic_mask_strategy=config.data.synthetic_mask_strategy,
        synthetic_mask_ratio=config.data.synthetic_mask_ratio,
        seed=config.seed,
    )

    logger.info("Train: %d samples", len(train_ds))
    logger.info("Val:   %d samples", len(val_ds))
    logger.info("Test:  %d samples", len(test_ds))

    # ── Build DataLoaders ──────────────────────────────────────────
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True,
    )

    # ── Select Trainer ─────────────────────────────────────────────
    model_name = config.model.name.lower()

    if model_name == "unet":
        from trainers.baseline_trainer import BaselineTrainer
        trainer = BaselineTrainer(
            config=config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            metadata=metadata,
        )
    elif model_name == "convlstm_unet":
        # Phase 2 — to be implemented
        raise NotImplementedError(
            "ConvLSTM U-Net trainer not yet implemented. "
            "Complete Phase 1 baseline first."
        )
    elif model_name == "gan":
        # Phase 3 — to be implemented
        raise NotImplementedError(
            "GAN trainer not yet implemented. "
            "Complete Phase 2 ConvLSTM baseline first."
        )
    else:
        logger.error("Unknown model: '%s'. Choose from: unet, convlstm_unet, gan", model_name)
        sys.exit(1)

    # ── Train ──────────────────────────────────────────────────────
    logger.info("Starting training with model: %s", model_name)
    history = trainer.train()

    # ── Save Visualizations ────────────────────────────────────────
    if config.output.save_visualizations:
        from visualization.loss_plots import plot_loss_curves, plot_metric_curves

        viz_dir = Path(config.output.output_dir) / "visualizations"
        viz_dir.mkdir(parents=True, exist_ok=True)

        plot_loss_curves(
            history["train_loss"],
            history["val_loss"],
            save_path=viz_dir / "loss_curves.png",
        )
        plot_metric_curves(
            {k: history[k] for k in ["rmse_gap", "mae_gap", "bias_gap", "corr_gap"]},
            save_path=viz_dir / "metric_curves.png",
        )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
