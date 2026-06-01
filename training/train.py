"""
SR-GAN Training Script for Wind-Speed Super-Resolution.

Two-stage training pipeline:
    Stage 1: Generator Pretraining (Masked L1 only, no discriminator)
    Stage 2: GAN Fine-Tuning (Generator + Discriminator, full loss)

Features (incorporating all review recommendations):
    - Resume training support (review #5: --resume)
    - CosineAnnealingLR scheduler (review #6)
    - AMP mixed precision
    - Gradient clipping
    - TensorBoard logging (review #12)
    - Best-metric checkpoints: RMSE, SSIM, PSNR (review #3)
    - Full-scene validation every epoch (review #2)
    - Early stopping

Usage:
    # Train from scratch
    python train.py --config configs/config.yaml

    # Resume from checkpoint
    python train.py --config configs/config.yaml --resume checkpoints/last_checkpoint.pth

    # Colab mode (auto-detect path)
    python train.py --config configs/config.yaml --colab
"""

import argparse
import os
import sys
import time
from typing import Dict, Optional

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.seed import set_seed
from utils.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    build_checkpoint_state,
    BestMetricTracker,
)
from utils.metrics import compute_all_metrics
from datasets.wind_dataset import WindSRDataset, get_temporal_split
from models.generator import SRResNet
from models.discriminator import PatchGANDiscriminator
from models.losses import GeneratorLoss, adversarial_loss_d, masked_l1_loss
from training.validate import validate


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_data_path(config: dict, colab: bool = False) -> str:
    """Resolve dataset path based on environment."""
    if colab:
        path = config["data"]["colab_path"]
    else:
        path = config["data"]["local_path"]

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    return path


def create_dataloaders(config: dict, data_path: str) -> tuple:
    """
    Create train, validation (patch + full-scene), and test datasets/loaders.

    Returns:
        (train_loader, val_loader, val_full_dataset, test_full_dataset,
         train_indices, val_indices, test_indices)
    """
    import netCDF4 as nc

    ds = nc.Dataset(data_path, "r")
    n_time = ds.variables[config["data"]["lr_variable"]].shape[0]
    ds.close()

    # Temporal split (review #10: no random splitting)
    train_idx, val_idx, test_idx = get_temporal_split(
        n_time,
        config["data"]["train_ratio"],
        config["data"]["val_ratio"],
        config["data"]["test_ratio"],
    )

    print(f"  Temporal split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Training dataset (patch mode)
    train_dataset = WindSRDataset(
        nc_path=data_path,
        lr_variable=config["data"]["lr_variable"],
        hr_variable=config["data"]["hr_variable"],
        mask_variable=config["data"]["mask_variable"],
        quality_variable=config["data"]["quality_variable"],
        time_indices=train_idx,
        mode="patch",
        lr_patch_size=config["patches"]["lr_patch_size"],
        hr_patch_size=config["patches"]["hr_patch_size"],
        min_ocean_fraction=config["patches"]["min_ocean_fraction"],
        patches_per_image=config["patches"]["patches_per_image"],
        scale_factor=config["model"]["scale_factor"],
        cache=config["data"]["cache_dataset"],
    )

    # Validation dataset (patch mode — for fast per-batch metrics)
    val_patch_dataset = WindSRDataset(
        nc_path=data_path,
        lr_variable=config["data"]["lr_variable"],
        hr_variable=config["data"]["hr_variable"],
        mask_variable=config["data"]["mask_variable"],
        quality_variable=config["data"]["quality_variable"],
        time_indices=val_idx,
        mode="patch",
        lr_patch_size=config["patches"]["lr_patch_size"],
        hr_patch_size=config["patches"]["hr_patch_size"],
        min_ocean_fraction=config["patches"]["min_ocean_fraction"],
        patches_per_image=config["patches"]["patches_per_image"],
        scale_factor=config["model"]["scale_factor"],
        cache=config["data"]["cache_dataset"],
    )

    # Validation dataset (full-scene mode — review #2)
    val_full_dataset = WindSRDataset(
        nc_path=data_path,
        lr_variable=config["data"]["lr_variable"],
        hr_variable=config["data"]["hr_variable"],
        mask_variable=config["data"]["mask_variable"],
        quality_variable=config["data"]["quality_variable"],
        time_indices=val_idx,
        mode="full",
        cache=config["data"]["cache_dataset"],
    )

    # Test dataset (full-scene mode)
    test_full_dataset = WindSRDataset(
        nc_path=data_path,
        lr_variable=config["data"]["lr_variable"],
        hr_variable=config["data"]["hr_variable"],
        mask_variable=config["data"]["mask_variable"],
        quality_variable=config["data"]["quality_variable"],
        time_indices=test_idx,
        mode="full",
        cache=config["data"]["cache_dataset"],
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_patch_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )

    return (
        train_loader,
        val_loader,
        val_full_dataset,
        test_full_dataset,
        train_idx,
        val_idx,
        test_idx,
    )


def create_models(config: dict, device: torch.device) -> tuple:
    """Create generator and discriminator models."""
    gen_cfg = config["model"]["generator"]
    disc_cfg = config["model"]["discriminator"]

    generator = SRResNet(
        in_channels=config["model"]["in_channels"],
        num_features=gen_cfg["num_features"],
        num_residual_blocks=gen_cfg["num_residual_blocks"],
        scale_factor=config["model"]["scale_factor"],
    ).to(device)

    discriminator = PatchGANDiscriminator(
        in_channels=config["model"]["in_channels"],
        base_channels=disc_cfg["base_channels"],
        use_spectral_norm=disc_cfg["spectral_norm"],
    ).to(device)

    # Print model sizes
    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"  Generator params:     {g_params:,}")
    print(f"  Discriminator params: {d_params:,}")

    return generator, discriminator


def train_pretrain(
    config: dict,
    generator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_full_dataset,
    device: torch.device,
    start_epoch: int = 0,
    best_metrics: Optional[Dict[str, float]] = None,
    writer=None,
) -> Dict[str, float]:
    """
    Stage 1: Generator Pretraining with Masked L1 loss only.

    Args:
        config: Configuration dictionary.
        generator: Generator model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (patches).
        val_full_dataset: Validation dataset (full scenes).
        device: Computation device.
        start_epoch: Starting epoch (for resume).
        best_metrics: Best metrics so far (for resume).
        writer: TensorBoard SummaryWriter.

    Returns:
        Best metrics dictionary.
    """
    print("\n" + "=" * 60)
    print("STAGE 1: Generator Pretraining (Masked L1)")
    print("=" * 60)

    opt_cfg = config["training"]["optimizer"]
    optimizer = optim.AdamW(
        generator.parameters(),
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        weight_decay=opt_cfg["weight_decay"],
    )

    # CosineAnnealingLR (review #6)
    total_epochs = config["training"]["pretrain_epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,
        eta_min=config["training"]["scheduler"]["eta_min"],
    )

    # AMP
    scaler = GradScaler(enabled=config["training"]["amp"] and torch.cuda.is_available())

    # Metric tracker (review #3)
    metric_tracker = BestMetricTracker(config["checkpoint"]["save_dir"])
    if best_metrics:
        metric_tracker.best.update(best_metrics)

    # Early stopping (Smoothed & Phase-aware)
    es_cfg = config["training"]["early_stopping"]
    best_smoothed_metric = float("inf") if es_cfg["mode"] == "min" else 0.0
    smoothed_metric = None
    patience_counter = 0
    alpha = 0.2  # EMA smoothing factor
    min_delta = 1e-4

    grad_clip = config["training"]["gradient_clip"]

    for epoch in range(start_epoch, total_epochs):
        generator.train()
        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{total_epochs}")
        for batch in pbar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            mask = batch["mask"].to(device)

            optimizer.zero_grad()

            with autocast(enabled=config["training"]["amp"] and torch.cuda.is_available()):
                sr = generator(lr)
                loss = masked_l1_loss(sr, hr, mask)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation (patch + full-scene)
        val_metrics = validate(
            generator, val_loader, val_full_dataset, device, num_full_scenes=5
        )
        val_rmse = val_metrics.get("rmse", float("inf"))

        # Log
        lr_current = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch+1}/{total_epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val RMSE: {val_rmse:.6f} | "
            f"LR: {lr_current:.2e}"
        )

        # TensorBoard
        if writer:
            writer.add_scalar("pretrain/train_loss", avg_train_loss, epoch)
            writer.add_scalar("pretrain/val_rmse", val_rmse, epoch)
            writer.add_scalar("pretrain/lr", lr_current, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"pretrain/{k}", v, epoch)

        # Best-metric checkpoints
        state = build_checkpoint_state(
            epoch, generator, None, optimizer, None, scheduler, None,
            metric_tracker.get_best(), "pretrain"
        )

        if val_metrics.get("rmse", None) is not None:
            metric_tracker.update("rmse", val_metrics["rmse"], state)
        if val_metrics.get("scene_ssim", None) is not None:
            metric_tracker.update("ssim", val_metrics["scene_ssim"], state)
        if val_metrics.get("scene_psnr", None) is not None:
            metric_tracker.update("psnr", val_metrics["scene_psnr"], state)

        # Save periodic checkpoint
        if (epoch + 1) % config["checkpoint"]["save_interval"] == 0:
            path = os.path.join(
                config["checkpoint"]["save_dir"],
                f"pretrain_epoch_{epoch+1}.pth",
            )
            save_checkpoint(state, path)

        # Save last checkpoint (for resume)
        save_checkpoint(
            state,
            os.path.join(config["checkpoint"]["save_dir"], "last_checkpoint.pth"),
        )

        # Global Minima Search: Smoothed Early Stopping
        # 1. EMA smoothing filters out local minima and noisy spikes
        # 2. Burn-in prevents stopping before CosineAnnealingLR drops
        if smoothed_metric is None:
            smoothed_metric = val_rmse
        else:
            smoothed_metric = alpha * val_rmse + (1 - alpha) * smoothed_metric

        improved = False
        if es_cfg["mode"] == "min":
            if smoothed_metric < best_smoothed_metric - min_delta:
                best_smoothed_metric = smoothed_metric
                improved = True
        else:
            if smoothed_metric > best_smoothed_metric + min_delta:
                best_smoothed_metric = smoothed_metric
                improved = True

        if improved:
            patience_counter = 0
        else:
            patience_counter += 1

        burn_in = int(total_epochs * 0.8)  # Require 80% completion for LR decay
        if patience_counter >= es_cfg["patience"] and epoch >= burn_in:
            print(f"  Early stopping at epoch {epoch+1} (smoothed_rmse={smoothed_metric:.6f})")
            break

    return metric_tracker.get_best()


def train_gan(
    config: dict,
    generator: nn.Module,
    discriminator: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_full_dataset,
    device: torch.device,
    start_epoch: int = 0,
    best_metrics: Optional[Dict[str, float]] = None,
    writer=None,
) -> Dict[str, float]:
    """
    Stage 2: GAN Fine-Tuning (Generator + Discriminator).

    Args:
        config: Configuration dictionary.
        generator: Generator model (pretrained).
        discriminator: Discriminator model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader (patches).
        val_full_dataset: Validation dataset (full scenes).
        device: Computation device.
        start_epoch: Starting epoch (for resume).
        best_metrics: Best metrics so far (for resume).
        writer: TensorBoard SummaryWriter.

    Returns:
        Best metrics dictionary.
    """
    print("\n" + "=" * 60)
    print("STAGE 2: GAN Fine-Tuning")
    print("=" * 60)

    opt_cfg = config["training"]["optimizer"]
    loss_cfg = config["loss"]

    optimizer_g = optim.AdamW(
        generator.parameters(),
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        weight_decay=opt_cfg["weight_decay"],
    )

    optimizer_d = optim.AdamW(
        discriminator.parameters(),
        lr=opt_cfg["lr"],
        betas=tuple(opt_cfg["betas"]),
        weight_decay=opt_cfg["weight_decay"],
    )

    total_epochs = config["training"]["gan_epochs"]

    scheduler_g = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_g,
        T_max=total_epochs,
        eta_min=config["training"]["scheduler"]["eta_min"],
    )
    scheduler_d = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_d,
        T_max=total_epochs,
        eta_min=config["training"]["scheduler"]["eta_min"],
    )

    gen_criterion = GeneratorLoss(
        pixel_weight=loss_cfg["pixel_weight"],
        adversarial_weight=loss_cfg["adversarial_weight"],
        gradient_weight=loss_cfg["gradient_weight"],
    )

    scaler = GradScaler(enabled=config["training"]["amp"] and torch.cuda.is_available())

    metric_tracker = BestMetricTracker(config["checkpoint"]["save_dir"])
    if best_metrics:
        metric_tracker.best.update(best_metrics)

    es_cfg = config["training"]["early_stopping"]
    best_smoothed_metric = float("inf")
    smoothed_metric = None
    patience_counter = 0
    alpha = 0.1  # Heavier smoothing for GANs
    min_delta = 1e-4
    grad_clip = config["training"]["gradient_clip"]

    for epoch in range(start_epoch, total_epochs):
        generator.train()
        discriminator.train()

        g_loss_sum = 0.0
        d_loss_sum = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"GAN Epoch {epoch+1}/{total_epochs}")
        for batch in pbar:
            lr = batch["lr"].to(device)
            hr = batch["hr"].to(device)
            mask = batch["mask"].to(device)

            # --- Train Discriminator ---
            optimizer_d.zero_grad()

            with autocast(enabled=config["training"]["amp"] and torch.cuda.is_available()):
                sr = generator(lr).detach()
                real_preds = discriminator(hr)
                fake_preds = discriminator(sr)
                d_loss = adversarial_loss_d(real_preds, fake_preds)

            scaler.scale(d_loss).backward()
            scaler.unscale_(optimizer_d)
            nn.utils.clip_grad_norm_(discriminator.parameters(), grad_clip)
            scaler.step(optimizer_d)
            scaler.update()

            # --- Train Generator ---
            optimizer_g.zero_grad()

            with autocast(enabled=config["training"]["amp"] and torch.cuda.is_available()):
                sr = generator(lr)
                fake_preds = discriminator(sr)
                g_losses = gen_criterion(sr, hr, mask, fake_preds)
                g_loss = g_losses["total_loss"]

            scaler.scale(g_loss).backward()
            scaler.unscale_(optimizer_g)
            nn.utils.clip_grad_norm_(generator.parameters(), grad_clip)
            scaler.step(optimizer_g)
            scaler.update()

            g_loss_sum += g_loss.item()
            d_loss_sum += d_loss.item()
            n_batches += 1

            pbar.set_postfix({
                "G": f"{g_loss.item():.4f}",
                "D": f"{d_loss.item():.4f}",
            })

        scheduler_g.step()
        scheduler_d.step()

        avg_g_loss = g_loss_sum / max(n_batches, 1)
        avg_d_loss = d_loss_sum / max(n_batches, 1)

        # Validation
        val_metrics = validate(
            generator, val_loader, val_full_dataset, device, num_full_scenes=5
        )
        val_rmse = val_metrics.get("rmse", float("inf"))

        lr_g = optimizer_g.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch+1}/{total_epochs} | "
            f"G Loss: {avg_g_loss:.6f} | D Loss: {avg_d_loss:.6f} | "
            f"Val RMSE: {val_rmse:.6f} | LR: {lr_g:.2e}"
        )

        # TensorBoard
        if writer:
            global_step = config["training"]["pretrain_epochs"] + epoch
            writer.add_scalar("gan/g_loss", avg_g_loss, global_step)
            writer.add_scalar("gan/d_loss", avg_d_loss, global_step)
            writer.add_scalar("gan/val_rmse", val_rmse, global_step)
            writer.add_scalar("gan/lr_g", lr_g, global_step)
            for k, v in val_metrics.items():
                writer.add_scalar(f"gan/{k}", v, global_step)

        # Best-metric checkpoints
        state = build_checkpoint_state(
            epoch, generator, discriminator, optimizer_g, optimizer_d,
            scheduler_g, scheduler_d, metric_tracker.get_best(), "gan"
        )

        if val_metrics.get("rmse", None) is not None:
            metric_tracker.update("rmse", val_metrics["rmse"], state)
        if val_metrics.get("scene_ssim", None) is not None:
            metric_tracker.update("ssim", val_metrics["scene_ssim"], state)
        if val_metrics.get("scene_psnr", None) is not None:
            metric_tracker.update("psnr", val_metrics["scene_psnr"], state)

        # Periodic checkpoint
        if (epoch + 1) % config["checkpoint"]["save_interval"] == 0:
            path = os.path.join(
                config["checkpoint"]["save_dir"],
                f"gan_epoch_{epoch+1}.pth",
            )
            save_checkpoint(state, path)

        # Last checkpoint
        save_checkpoint(
            state,
            os.path.join(config["checkpoint"]["save_dir"], "last_checkpoint.pth"),
        )

        # Save discriminator best
        d_path = os.path.join(config["checkpoint"]["save_dir"], "best_discriminator.pth")
        save_checkpoint(
            {"discriminator_state_dict": discriminator.state_dict(), "epoch": epoch},
            d_path,
        )

        # Global Minima Search: Smoothed Early Stopping for GAN
        # 1. Heavy EMA smoothing filters out extreme adversarial oscillations
        # 2. Burn-in extended to 80% to ensure LR decay reaches the global basin
        if smoothed_metric is None:
            smoothed_metric = val_rmse
        else:
            smoothed_metric = alpha * val_rmse + (1 - alpha) * smoothed_metric

        if smoothed_metric < best_smoothed_metric - min_delta:
            best_smoothed_metric = smoothed_metric
            patience_counter = 0
        else:
            patience_counter += 1

        burn_in = int(total_epochs * 0.8)
        if patience_counter >= es_cfg["patience"] and epoch >= burn_in:
            print(f"  Early stopping at epoch {epoch+1} (smoothed_rmse={smoothed_metric:.6f})")
            break

    return metric_tracker.get_best()


def main():
    parser = argparse.ArgumentParser(description="SR-GAN Wind Speed Training")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to configuration YAML file.",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint for resume training (review #5).",
    )
    parser.add_argument(
        "--colab", action="store_true",
        help="Use Google Colab data path.",
    )
    parser.add_argument(
        "--stage", type=str, default="all", choices=["pretrain", "gan", "all"],
        help="Training stage to run.",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Route outputs to Google Drive in Colab mode
    if args.colab:
        base_dir = "/content/drive/MyDrive/SR-GAN_Wind"
        config["checkpoint"]["save_dir"] = os.path.join(base_dir, config["checkpoint"]["save_dir"])
        config["logging"]["tensorboard_dir"] = os.path.join(base_dir, config["logging"]["tensorboard_dir"])
        print(f"  Colab Mode: Routing outputs to {base_dir}")

    # Set seed
    set_seed(config["seed"], config["deterministic"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'=' * 60}")
    print(f"SR-GAN Wind Speed Super-Resolution Training")
    print(f"{'=' * 60}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Data path
    data_path = get_data_path(config, colab=args.colab)
    print(f"  Dataset: {data_path}")

    # Create data
    print("\nCreating datasets...")
    (
        train_loader,
        val_loader,
        val_full_dataset,
        test_full_dataset,
        train_idx,
        val_idx,
        test_idx,
    ) = create_dataloaders(config, data_path)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")
    print(f"  Val scenes:    {len(val_full_dataset)}")
    print(f"  Test scenes:   {len(test_full_dataset)}")

    # Create models
    print("\nCreating models...")
    generator, discriminator = create_models(config, device)

    # TensorBoard (review #12)
    writer = None
    if config["logging"]["tensorboard"]:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = config["logging"]["tensorboard_dir"]
            os.makedirs(tb_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=tb_dir)
            print(f"  TensorBoard: {tb_dir}")
        except ImportError:
            print("  TensorBoard not available, skipping.")

    # Resume support (review #5)
    start_epoch_pretrain = 0
    start_epoch_gan = 0
    best_metrics = None
    resume_stage = None

    if args.resume:
        print(f"\nResuming from: {args.resume}")
        checkpoint = load_checkpoint(args.resume, device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        best_metrics = checkpoint.get("best_metrics", None)
        resume_stage = checkpoint.get("stage", "pretrain")
        resume_epoch = checkpoint.get("epoch", 0) + 1
        print(f"  Resumed stage: {resume_stage}, epoch: {resume_epoch}")

        if resume_stage == "pretrain":
            start_epoch_pretrain = resume_epoch
        elif resume_stage == "gan":
            start_epoch_gan = resume_epoch
            if "discriminator_state_dict" in checkpoint:
                discriminator.load_state_dict(checkpoint["discriminator_state_dict"])

    # === TRAINING ===
    os.makedirs(config["checkpoint"]["save_dir"], exist_ok=True)

    if args.stage in ("pretrain", "all"):
        if resume_stage != "gan":  # Don't re-run pretrain if resuming GAN
            best_metrics = train_pretrain(
                config, generator, train_loader, val_loader,
                val_full_dataset, device, start_epoch_pretrain,
                best_metrics, writer,
            )

    if args.stage in ("gan", "all"):
        # Load best pretrained generator if starting GAN fresh
        if args.stage == "all" and start_epoch_gan == 0:
            best_gen_path = os.path.join(
                config["checkpoint"]["save_dir"], "best_rmse_generator.pth"
            )
            if os.path.isfile(best_gen_path):
                print(f"\nLoading best pretrained generator: {best_gen_path}")
                ckpt = load_checkpoint(best_gen_path, device)
                generator.load_state_dict(ckpt["generator_state_dict"])

        best_metrics = train_gan(
            config, generator, discriminator, train_loader, val_loader,
            val_full_dataset, device, start_epoch_gan,
            best_metrics, writer,
        )

    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    if best_metrics:
        print("  Best Metrics:")
        for k, v in best_metrics.items():
            print(f"    {k}: {v:.6f}")

    if writer:
        writer.close()

    print("\nCheckpoints saved to:", config["checkpoint"]["save_dir"])


if __name__ == "__main__":
    main()
