"""
GAN trainer for WindGapGAN — Full adversarial training loop.

Trains a MaskedUNet generator against a PatchGAN discriminator with:
    - LSGAN (least-squares) adversarial loss for stability
    - Pixel-wise L1 loss on gap regions (MaskedL1Loss)
    - Perceptual loss (discriminator feature matching)
    - Spectral loss (FFT frequency matching)
    - Spatial gradient loss (boundary smoothness)
    - Adversarial weight ramp (warm up generator first)

Two-timescale training:
    Generator LR:     1e-4  (default)
    Discriminator LR:  4e-4  (faster to catch up)

Key design decisions:
    - Generator uses hard masked merge during inference, but soft mode
      during training for better gradient flow through gap regions.
    - Discriminator sees [masked_input, mask, image] — conditional GAN.
    - Feature matching from layers 1-4 of discriminator.
    - Adversarial loss is ramped linearly over first N epochs.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from losses.adversarial import AdversarialLoss
from losses.gradient_loss import GradientLoss
from losses.masked_l1 import MaskedL1Loss
from losses.perceptual_loss import PerceptualLoss
from losses.spectral_loss import SpectralLoss
from models.discriminator import PatchGANDiscriminator
from models.unet import MaskedUNet
from trainers.base_trainer import BaseTrainer
from utils.config import Config

logger = logging.getLogger(__name__)


class GANTrainer(BaseTrainer):
    """
    Full adversarial trainer for gap-filling.

    Alternates between generator and discriminator updates with
    a comprehensive loss composition designed for producing realistic
    wind field reconstructions.
    """

    def __init__(
        self,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        # Build discriminator before calling super().__init__
        # (which calls _build_model and _build_optimizer)
        self._disc_built = False
        super().__init__(config, train_loader, val_loader, device, metadata)

        # ── Loss Functions ─────────────────────────────────────────
        self.pixel_criterion = MaskedL1Loss(observed_weight=0.01)
        self.adv_criterion = AdversarialLoss(mode="lsgan")
        self.perceptual_criterion = PerceptualLoss(
            n_layers=self.discriminator.n_layers + 1
        )
        self.spectral_criterion = SpectralLoss(log_scale=True)
        self.gradient_criterion = (
            GradientLoss()
            if config.training.gradient_loss_weight > 0
            else None
        )

        # ── Loss Weights ───────────────────────────────────────────
        self.pixel_weight = config.training.pixel_loss_weight
        self.adv_weight = config.training.adversarial_weight
        self.adv_ramp_epochs = config.training.adversarial_ramp_epochs
        self.perceptual_weight = config.training.perceptual_loss_weight
        self.spectral_weight = config.training.spectral_loss_weight
        self.gradient_weight = config.training.gradient_loss_weight

        # ── Extended History ───────────────────────────────────────
        self.history.update({
            "g_loss_total": [],
            "g_loss_pixel": [],
            "g_loss_adv": [],
            "g_loss_perceptual": [],
            "g_loss_spectral": [],
            "g_loss_gradient": [],
            "d_loss": [],
            "d_real_score": [],
            "d_fake_score": [],
        })

        # Track current epoch for ramp
        self._current_epoch = 0

    def _build_model(self) -> nn.Module:
        """Build generator (MaskedUNet) and discriminator (PatchGAN)."""
        cfg = self.config.model

        # Generator: use soft merge during training for gradient flow
        generator = MaskedUNet(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            base_features=cfg.base_features,
            depth=cfg.depth,
            dropout=cfg.dropout,
            use_batch_norm=cfg.use_batch_norm,
            use_attention=cfg.use_attention,
            use_tanh=cfg.use_tanh,
            hard_merge=False,  # Soft merge during training
        )

        # Discriminator: conditional PatchGAN
        # Input = [masked_field(1) + mask(1) + image(1)] = 3 channels
        disc_in_channels = 3
        self.discriminator = PatchGANDiscriminator(
            in_channels=disc_in_channels,
            base_features=cfg.disc_base_features,
            n_layers=cfg.disc_depth,
            use_spectral_norm=cfg.use_spectral_norm,
        )
        self.discriminator.to(self.config.resolve_device())
        self._disc_built = True

        return generator

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build separate optimizers for generator and discriminator."""
        cfg = self.config.training

        # Generator optimizer
        self.optimizer_g = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr_generator,
            betas=(0.5, 0.999),
            weight_decay=cfg.weight_decay,
        )

        # Discriminator optimizer
        self.optimizer_d = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=cfg.lr_discriminator,
            betas=(0.5, 0.999),
            weight_decay=cfg.weight_decay,
        )

        # Discriminator scaler for AMP
        self.scaler_d = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # Return generator optimizer as the "main" optimizer
        # (base_trainer uses self.optimizer for scheduler)
        return self.optimizer_g

    def _get_adv_weight(self) -> float:
        """Get current adversarial weight with linear ramp."""
        if self.adv_ramp_epochs <= 0:
            return self.adv_weight
        ramp = min(1.0, self._current_epoch / self.adv_ramp_epochs)
        return self.adv_weight * ramp

    def _train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Execute one GAN training step.

        Steps:
            1. Generate fake images
            2. Train discriminator (real vs fake)
            3. Train generator (fool discriminator + reconstruction losses)
        """
        inputs = batch["input"].to(self.device, non_blocking=True)
        targets = batch["target"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        land_masks = batch["land_mask"].to(self.device, non_blocking=True)

        # Extract components for discriminator conditioning
        # inputs shape: (B, T, 2, H, W) → fold time
        B_orig = inputs.shape[0]
        has_time = inputs.ndim == 5
        if has_time:
            B, T, C, H, W = inputs.shape
            inputs_flat = inputs.reshape(B * T, C, H, W)
            targets_flat = targets.reshape(B * T, 1, H, W)
            masks_flat = masks.reshape(B * T, 1, H, W)
        else:
            inputs_flat = inputs
            targets_flat = targets
            masks_flat = masks

        masked_field = inputs_flat[:, 0:1, :, :]  # (B*T, 1, H, W)
        mask_channel = inputs_flat[:, 1:2, :, :]   # (B*T, 1, H, W)

        # ══════════════════════════════════════════════════════════════
        # Step 1: Generate fake images
        # ══════════════════════════════════════════════════════════════
        with torch.amp.autocast('cuda', enabled=self.use_amp):
            fake_raw = self.model(inputs)  # soft merge → raw prediction

        # For loss computation and discriminator, apply hard merge manually
        if has_time:
            fake_flat = fake_raw.reshape(B * T, 1, H, W)
        else:
            fake_flat = fake_raw

        # Hard merge for the output used in loss computation:
        # observed pixels come from input, gap pixels from prediction
        fake_merged = mask_channel * masked_field + (1.0 - mask_channel) * fake_flat

        # ══════════════════════════════════════════════════════════════
        # Step 2: Train Discriminator
        # ══════════════════════════════════════════════════════════════
        self.optimizer_d.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # Real: discriminator sees [condition, ground_truth]
            real_input = PatchGANDiscriminator.build_input(
                masked_field, mask_channel, targets_flat,
            )
            disc_real = self.discriminator(real_input)

            # Fake: discriminator sees [condition, fake_merged]
            fake_input = PatchGANDiscriminator.build_input(
                masked_field, mask_channel, fake_merged.detach(),
            )
            disc_fake = self.discriminator(fake_input)

            d_loss = self.adv_criterion.discriminator_loss(disc_real, disc_fake)

        self.scaler_d.scale(d_loss).backward()

        if self.config.training.grad_clip > 0:
            self.scaler_d.unscale_(self.optimizer_d)
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                self.config.training.grad_clip,
            )

        self.scaler_d.step(self.optimizer_d)
        self.scaler_d.update()

        # ══════════════════════════════════════════════════════════════
        # Step 3: Train Generator
        # ══════════════════════════════════════════════════════════════
        self.optimizer_g.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            # ── Pixel Loss (L1 on gaps) ───────────────────────────
            pixel_loss = self.pixel_criterion(
                fake_merged, targets_flat, masks_flat, land_mask=land_masks,
            )

            # ── Adversarial Loss ──────────────────────────────────
            fake_input_g = PatchGANDiscriminator.build_input(
                masked_field, mask_channel, fake_merged,
            )
            disc_fake_g, fake_features = self.discriminator(
                fake_input_g, return_features=True,
            )
            adv_loss = self.adv_criterion.generator_loss(disc_fake_g)

            # ── Perceptual Loss (feature matching) ────────────────
            with torch.no_grad():
                real_input_g = PatchGANDiscriminator.build_input(
                    masked_field, mask_channel, targets_flat,
                )
                _, real_features = self.discriminator(
                    real_input_g, return_features=True,
                )
            perc_loss = self.perceptual_criterion(fake_features, real_features)

            # ── Spectral Loss ─────────────────────────────────────
            spec_loss = self.spectral_criterion(fake_merged, targets_flat)

            # ── Gradient Loss ─────────────────────────────────────
            if self.gradient_criterion is not None:
                grad_loss = self.gradient_criterion(fake_merged, targets_flat)
            else:
                grad_loss = torch.tensor(0.0, device=self.device)

            # ── Total Generator Loss ──────────────────────────────
            current_adv_w = self._get_adv_weight()
            g_loss = (
                self.pixel_weight * pixel_loss
                + current_adv_w * adv_loss
                + self.perceptual_weight * perc_loss
                + self.spectral_weight * spec_loss
                + self.gradient_weight * grad_loss
            )

        self.scaler.scale(g_loss).backward()

        # Compute generator gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5

        if self.config.training.grad_clip > 0:
            self.scaler.unscale_(self.optimizer_g)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip,
            )

        self.scaler.step(self.optimizer_g)
        self.scaler.update()

        return {
            "loss": g_loss.item(),
            "grad_norm": total_norm,
            "g_loss_pixel": pixel_loss.item(),
            "g_loss_adv": adv_loss.item(),
            "g_loss_perceptual": perc_loss.item(),
            "g_loss_spectral": spec_loss.item(),
            "g_loss_gradient": grad_loss.item() if isinstance(grad_loss, torch.Tensor) else 0.0,
            "d_loss": d_loss.item(),
            "d_real_score": disc_real.mean().item(),
            "d_fake_score": disc_fake.mean().item(),
        }

    def _val_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Validation step — uses hard masked merge for proper evaluation.

        During validation we apply the hard constraint (preserve observed
        pixels) and compute gap-only metrics.
        """
        inputs = batch["input"].to(self.device, non_blocking=True)
        targets = batch["target"].to(self.device, non_blocking=True)
        masks = batch["mask"].to(self.device, non_blocking=True)
        land_masks = batch["land_mask"].to(self.device, non_blocking=True)

        # Fold time dimension
        has_time = inputs.ndim == 5
        if has_time:
            B, T, C, H, W = inputs.shape
            inputs_flat = inputs.reshape(B * T, C, H, W)
            targets_flat = targets.reshape(B * T, 1, H, W)
            masks_flat = masks.reshape(B * T, 1, H, W)
        else:
            inputs_flat = inputs
            targets_flat = targets
            masks_flat = masks

        masked_field = inputs_flat[:, 0:1, :, :]
        mask_channel = inputs_flat[:, 1:2, :, :]

        with torch.amp.autocast('cuda', enabled=self.use_amp):
            fake_raw = self.model(inputs)
            if has_time:
                fake_flat = fake_raw.reshape(B * T, 1, H, W)
            else:
                fake_flat = fake_raw

            # Hard merge for evaluation: preserve observed pixels
            predictions = mask_channel * masked_field + (1.0 - mask_channel) * fake_flat

            loss = self.pixel_criterion(predictions, targets_flat, masks_flat, land_mask=land_masks)

        # Gap-only metrics (ocean gaps, excluding land)
        ocean_gap_mask = ((1.0 - masks_flat) * land_masks).bool()
        metrics = {"loss": loss.item()}

        if ocean_gap_mask.any():
            gap_pred = predictions[ocean_gap_mask]
            gap_target = targets_flat[ocean_gap_mask]
            diff = gap_pred - gap_target

            metrics["rmse_gap"] = float(torch.sqrt(torch.mean(diff ** 2)).item())
            metrics["mae_gap"] = float(torch.mean(torch.abs(diff)).item())
            metrics["bias_gap"] = float(torch.mean(diff).item())

            if len(gap_pred) > 1:
                gp = gap_pred.detach().cpu().numpy()
                gt = gap_target.detach().cpu().numpy()
                if np.std(gp) > 1e-10 and np.std(gt) > 1e-10:
                    metrics["corr_gap"] = float(np.corrcoef(gp, gt)[0, 1])
                else:
                    metrics["corr_gap"] = float("nan")
            else:
                metrics["corr_gap"] = float("nan")
        else:
            metrics["rmse_gap"] = float("nan")
            metrics["mae_gap"] = float("nan")
            metrics["bias_gap"] = float("nan")
            metrics["corr_gap"] = float("nan")

        return metrics

    def train(self) -> dict[str, list[float]]:
        """
        Override train loop to track epoch number for adversarial ramp
        and handle G/D loss logging.
        """
        cfg = self.config.training
        logger.info("Starting GAN training: %d epochs, batch_size=%d", cfg.epochs, cfg.batch_size)
        logger.info(
            "Generator params: %s | Discriminator params: %s",
            f"{sum(p.numel() for p in self.model.parameters()):,}",
            f"{sum(p.numel() for p in self.discriminator.parameters()):,}",
        )
        logger.info(
            "Loss weights: pixel=%.1f, adv=%.3f (ramp %d ep), perc=%.1f, spec=%.1f, grad=%.1f",
            self.pixel_weight, self.adv_weight, self.adv_ramp_epochs,
            self.perceptual_weight, self.spectral_weight, self.gradient_weight,
        )

        import time
        from pathlib import Path
        from utils.convergence import ConvergenceDetector

        for epoch in range(self.start_epoch, cfg.epochs):
            self._current_epoch = epoch
            epoch_start = time.time()

            # Reshuffle training data
            if hasattr(self.train_loader.dataset, 'reshuffle'):
                self.train_loader.dataset.reshuffle()

            # ── Train Phase ────────────────────────────────────────
            self.model.train()
            self.discriminator.train()
            train_step_results = []

            for batch in self.train_loader:
                step_result = self._train_step(batch)
                train_step_results.append(step_result)

            # Average metrics
            avg_train_loss = float(np.mean([r["loss"] for r in train_step_results]))
            avg_d_loss = float(np.mean([r["d_loss"] for r in train_step_results]))
            avg_d_real = float(np.mean([r["d_real_score"] for r in train_step_results]))
            avg_d_fake = float(np.mean([r["d_fake_score"] for r in train_step_results]))

            # ── Validation Phase ───────────────────────────────────
            self.model.eval()
            self.discriminator.eval()
            val_metrics_list = []

            with torch.no_grad():
                for batch in self.val_loader:
                    step_result = self._val_step(batch)
                    val_metrics_list.append(step_result)

            avg_val = self._average_metrics(val_metrics_list)
            avg_val_loss = avg_val.get("loss", float("inf"))

            # ── Learning Rate ──────────────────────────────────────
            current_lr = self.optimizer_g.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            # ── Record History ─────────────────────────────────────
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["learning_rate"].append(current_lr)
            self.history["d_loss"].append(avg_d_loss)
            self.history["d_real_score"].append(avg_d_real)
            self.history["d_fake_score"].append(avg_d_fake)

            for key in ["rmse_gap", "mae_gap", "bias_gap", "corr_gap"]:
                self.history[key].append(avg_val.get(key, float("nan")))

            # Average per-component losses
            for key in ["g_loss_pixel", "g_loss_adv", "g_loss_perceptual",
                         "g_loss_spectral", "g_loss_gradient"]:
                avg_val_key = float(np.mean([r.get(key, 0.0) for r in train_step_results]))
                self.history[key].append(avg_val_key)

            avg_grad_norm = float(np.mean([r.get("grad_norm", 0.0) for r in train_step_results]))
            self.history["grad_norm"].append(avg_grad_norm)

            # ── Checkpointing ──────────────────────────────────────
            state = {
                "model_state_dict": self.model.state_dict(),
                "discriminator_state_dict": self.discriminator.state_dict(),
                "optimizer_g_state_dict": self.optimizer_g.state_dict(),
                "optimizer_d_state_dict": self.optimizer_d.state_dict(),
                "epoch": epoch,
                "config": self._config_to_dict(),
                "norm_stats": self.metadata.get("norm_stats", {}),
            }
            self.ckpt_manager.save(state, epoch, avg_val)

            # ── Logging ────────────────────────────────────────────
            elapsed = time.time() - epoch_start
            adv_w = self._get_adv_weight()
            logger.info(
                "Epoch %d/%d │ G: %.4f (px:%.4f adv:%.4f×%.2f prc:%.4f sp:%.4f) │ "
                "D: %.4f (R:%.2f/F:%.2f) │ RMSE: %.4f │ Corr: %.4f │ "
                "LR: %.2e │ %.1fs",
                epoch + 1, cfg.epochs,
                avg_train_loss,
                float(np.mean([r["g_loss_pixel"] for r in train_step_results])),
                float(np.mean([r["g_loss_adv"] for r in train_step_results])),
                adv_w,
                float(np.mean([r["g_loss_perceptual"] for r in train_step_results])),
                float(np.mean([r["g_loss_spectral"] for r in train_step_results])),
                avg_d_loss, avg_d_real, avg_d_fake,
                avg_val.get("rmse_gap", float("nan")),
                avg_val.get("corr_gap", float("nan")),
                current_lr, elapsed,
            )

            # TensorBoard logging
            if self.tb_writer:
                self.tb_writer.add_scalar("Loss/G_total", avg_train_loss, epoch)
                self.tb_writer.add_scalar("Loss/D_total", avg_d_loss, epoch)
                self.tb_writer.add_scalar("Loss/val", avg_val_loss, epoch)
                self.tb_writer.add_scalar("D/real_score", avg_d_real, epoch)
                self.tb_writer.add_scalar("D/fake_score", avg_d_fake, epoch)
                self.tb_writer.add_scalar("LR", current_lr, epoch)
                for key in ["rmse_gap", "mae_gap", "bias_gap", "corr_gap"]:
                    if key in avg_val:
                        self.tb_writer.add_scalar(f"Metrics/{key}", avg_val[key], epoch)

            # ── Convergence Detection ──────────────────────────────
            level, just_promoted = self.convergence_detector.step(
                epoch=epoch, val_loss=avg_val_loss, grad_norm=avg_grad_norm
            )
            if just_promoted:
                plateau_path = self.ckpt_manager.checkpoint_dir / f"plateau_level_{level}_epoch_{epoch + 1}.pt"
                torch.save(state, plateau_path)
                logger.info("Saved plateau checkpoint to %s", plateau_path)

                if level <= self.config.training.convergence.max_plateau_cycles:
                    logger.info("Plateau detected (Level %d). Continuing training.", level)
                else:
                    logger.info("True convergence reached. Stopping.")
                    break

        # Save history
        self._save_history()
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()

        logger.info("GAN Training complete.")
        return self.history
