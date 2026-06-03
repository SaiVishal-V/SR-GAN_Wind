"""
Baseline trainer for Phase 1: Masked U-Net.

Concrete trainer implementing:
    - MaskedUNet model
    - MaskedL1Loss
    - Standard train/val steps with gap-only metric computation
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from losses.masked_l1 import MaskedL1Loss
from losses.gradient_loss import GradientLoss
from models.unet import MaskedUNet
from trainers.base_trainer import BaseTrainer
from utils.config import Config

logger = logging.getLogger(__name__)


class BaselineTrainer(BaseTrainer):
    """
    Phase 1 trainer: Masked U-Net with Masked L1 loss.

    Train step:
        1. Forward pass through MaskedUNet
        2. Compute MaskedL1Loss on gap pixels
        3. Optionally add GradientLoss
        4. Backward + optimizer step

    Val step:
        1. Forward pass
        2. Compute gap-only metrics (RMSE, MAE, Bias, Correlation)
    """

    def __init__(
        self,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(config, train_loader, val_loader, device, metadata)

        # Loss functions
        self.criterion = MaskedL1Loss(observed_weight=0.01)
        self.gradient_loss = GradientLoss() if config.training.gradient_loss_weight > 0 else None
        self.gradient_weight = config.training.gradient_loss_weight

    def _build_model(self) -> nn.Module:
        """Build Masked U-Net model."""
        cfg = self.config.model
        model = MaskedUNet(
            in_channels=cfg.in_channels,
            out_channels=cfg.out_channels,
            base_features=cfg.base_features,
            depth=cfg.depth,
            dropout=cfg.dropout,
            use_batch_norm=cfg.use_batch_norm,
        )
        return model

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer."""
        cfg = self.config.training
        if cfg.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        elif cfg.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

    def _train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute one training step."""
        inputs = batch["input"].to(self.device)   # (B, T, 2, H, W)
        targets = batch["target"].to(self.device)  # (B, T, 1, H, W)
        masks = batch["mask"].to(self.device)      # (B, T, 1, H, W)

        # Forward
        predictions = self.model(inputs)  # (B, T, 1, H, W)

        # Masked L1 loss
        loss = self.criterion(predictions, targets, masks)

        # Optional gradient loss
        if self.gradient_loss is not None and self.gradient_weight > 0:
            g_loss = self.gradient_loss(predictions, targets)
            loss = loss + self.gradient_weight * g_loss

        # Backward
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.grad_clip,
            )

        self.optimizer.step()

        return {"loss": loss.item()}

    def _val_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute one validation step with gap-only metrics."""
        inputs = batch["input"].to(self.device)
        targets = batch["target"].to(self.device)
        masks = batch["mask"].to(self.device)

        predictions = self.model(inputs)

        # Loss
        loss = self.criterion(predictions, targets, masks)

        # Compute gap-only metrics
        pred_np = predictions.cpu().numpy().flatten()
        target_np = targets.cpu().numpy().flatten()
        mask_np = masks.cpu().numpy().flatten()

        gap_mask = (1.0 - mask_np).astype(bool)

        metrics = {"loss": loss.item()}

        if gap_mask.any():
            gap_pred = pred_np[gap_mask]
            gap_target = target_np[gap_mask]
            diff = gap_pred - gap_target

            metrics["rmse_gap"] = float(np.sqrt(np.mean(diff ** 2)))
            metrics["mae_gap"] = float(np.mean(np.abs(diff)))
            metrics["bias_gap"] = float(np.mean(diff))

            if len(gap_pred) > 1 and np.std(gap_pred) > 1e-10 and np.std(gap_target) > 1e-10:
                metrics["corr_gap"] = float(np.corrcoef(gap_pred, gap_target)[0, 1])
            else:
                metrics["corr_gap"] = float("nan")
        else:
            metrics["rmse_gap"] = float("nan")
            metrics["mae_gap"] = float("nan")
            metrics["bias_gap"] = float("nan")
            metrics["corr_gap"] = float("nan")

        return metrics
