"""
Base trainer for WindGapGAN.

Abstract base class providing:
    - Epoch loop with train/val phases
    - Metric tracking (loss, RMSE, MAE, Bias, Correlation — gap-only)
    - Checkpointing (latest, best-RMSE, best-MAE, best-Correlation)
    - TensorBoard + optional W&B logging
    - Learning rate scheduling with warmup
    - Gradient clipping
    - Early stopping
    - Config serialization per run
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.checkpoint import CheckpointManager
from utils.config import Config, save_config

logger = logging.getLogger(__name__)


class BaseTrainer(ABC):
    """
    Abstract base trainer for all WindGapGAN training phases.

    Subclasses must implement:
        - _build_model()
        - _build_optimizer()
        - _train_step(batch) → loss_dict
        - _val_step(batch) → metrics_dict
    """

    def __init__(
        self,
        config: Config,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.metadata = metadata or {}

        # Build model and optimizer (subclass responsibility)
        self.model = self._build_model()
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Output directories
        self.output_dir = Path(config.output.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint manager
        ckpt_cfg = config.checkpointing
        self.ckpt_manager = CheckpointManager(
            checkpoint_dir=ckpt_cfg.checkpoint_dir,
            save_best_rmse=ckpt_cfg.save_best_rmse,
            save_best_mae=ckpt_cfg.save_best_mae,
            save_best_corr=ckpt_cfg.save_best_corr,
            save_every=ckpt_cfg.save_every,
        )

        # History
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "rmse_gap": [],
            "mae_gap": [],
            "bias_gap": [],
            "corr_gap": [],
            "learning_rate": [],
        }

        # Early stopping
        self.patience = config.training.early_stopping_patience
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # TensorBoard
        self.tb_writer = None
        if config.logging.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=config.logging.log_dir)
            except ImportError:
                logger.warning("TensorBoard not available. Install tensorboard.")

        # W&B
        self.wandb_run = None
        if config.logging.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.logging.wandb_project,
                    entity=config.logging.wandb_entity,
                    config=self._config_to_dict(),
                )
            except ImportError:
                logger.warning("W&B not available. Install wandb.")

        # Save config for reproducibility
        save_config(config, self.output_dir / "config.yaml")

    @abstractmethod
    def _build_model(self) -> nn.Module:
        """Build and return the model."""

    @abstractmethod
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build and return the optimizer."""

    @abstractmethod
    def _train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Execute one training step.

        Args:
            batch: Dict with 'input', 'target', 'mask' tensors.

        Returns:
            Dict with at least 'loss' key.
        """

    @abstractmethod
    def _val_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """
        Execute one validation step.

        Returns:
            Dict with metrics: 'loss', 'rmse_gap', 'mae_gap', 'bias_gap', 'corr_gap'.
        """

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler."""
        cfg = self.config.training
        if cfg.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=cfg.epochs, eta_min=1e-7
            )
        elif cfg.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.5
            )
        elif cfg.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=10
            )
        elif cfg.scheduler == "none":
            return None
        else:
            logger.warning("Unknown scheduler '%s'. Using none.", cfg.scheduler)
            return None

    def train(self) -> dict[str, list[float]]:
        """
        Main training loop.

        Returns:
            Training history dict.
        """
        cfg = self.config.training
        logger.info("Starting training: %d epochs, batch_size=%d", cfg.epochs, cfg.batch_size)
        logger.info("Model parameters: %s", f"{sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(cfg.epochs):
            epoch_start = time.time()

            # ── Train Phase ────────────────────────────────────────
            self.model.train()
            train_losses = []

            for batch in self.train_loader:
                step_result = self._train_step(batch)
                train_losses.append(step_result["loss"])

            avg_train_loss = float(np.mean(train_losses))

            # ── Validation Phase ───────────────────────────────────
            self.model.eval()
            val_metrics_list: list[dict[str, float]] = []

            with torch.no_grad():
                for batch in self.val_loader:
                    step_result = self._val_step(batch)
                    val_metrics_list.append(step_result)

            # Average validation metrics
            avg_val = self._average_metrics(val_metrics_list)
            avg_val_loss = avg_val.get("loss", float("inf"))

            # ── Learning Rate ──────────────────────────────────────
            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(avg_val_loss)
                else:
                    self.scheduler.step()

            # ── Record History ─────────────────────────────────────
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)
            self.history["learning_rate"].append(current_lr)
            for key in ["rmse_gap", "mae_gap", "bias_gap", "corr_gap"]:
                self.history[key].append(avg_val.get(key, float("nan")))

            # ── Checkpointing ──────────────────────────────────────
            state = {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self._config_to_dict(),
                "norm_stats": self.metadata.get("norm_stats", {}),
            }
            self.ckpt_manager.save(state, epoch, avg_val)

            # ── Logging ────────────────────────────────────────────
            elapsed = time.time() - epoch_start
            logger.info(
                "Epoch %d/%d │ Train Loss: %.6f │ Val Loss: %.6f │ "
                "RMSE_gap: %.4f │ MAE_gap: %.4f │ Corr_gap: %.4f │ "
                "LR: %.2e │ %.1fs",
                epoch + 1,
                cfg.epochs,
                avg_train_loss,
                avg_val_loss,
                avg_val.get("rmse_gap", float("nan")),
                avg_val.get("mae_gap", float("nan")),
                avg_val.get("corr_gap", float("nan")),
                current_lr,
                elapsed,
            )

            # TensorBoard logging
            if self.tb_writer:
                self.tb_writer.add_scalar("Loss/train", avg_train_loss, epoch)
                self.tb_writer.add_scalar("Loss/val", avg_val_loss, epoch)
                self.tb_writer.add_scalar("LR", current_lr, epoch)
                for key in ["rmse_gap", "mae_gap", "bias_gap", "corr_gap"]:
                    if key in avg_val:
                        self.tb_writer.add_scalar(f"Metrics/{key}", avg_val[key], epoch)

            # W&B logging
            if self.wandb_run:
                import wandb
                log_dict = {
                    "epoch": epoch,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "learning_rate": current_lr,
                }
                log_dict.update({k: v for k, v in avg_val.items() if not np.isnan(v)})
                wandb.log(log_dict)

            # ── Early Stopping ─────────────────────────────────────
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if self.patience > 0 and self.epochs_without_improvement >= self.patience:
                logger.info(
                    "Early stopping triggered at epoch %d (no improvement for %d epochs)",
                    epoch + 1,
                    self.patience,
                )
                break

        # ── Save Training History ──────────────────────────────────
        self._save_history()

        # Close loggers
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            import wandb
            wandb.finish()

        logger.info("Training complete. Best val loss: %.6f", self.best_val_loss)
        return self.history

    def _average_metrics(self, metrics_list: list[dict[str, float]]) -> dict[str, float]:
        """Average a list of metric dicts."""
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        averaged = {}
        for key in keys:
            values = [m[key] for m in metrics_list if not np.isnan(m.get(key, float("nan")))]
            averaged[key] = float(np.mean(values)) if values else float("nan")
        return averaged

    def _save_history(self) -> None:
        """Save training history to CSV."""
        import pandas as pd

        df = pd.DataFrame(self.history)
        csv_path = self.output_dir / "metrics.csv"
        df.to_csv(csv_path, index_label="epoch", float_format="%.6f")
        logger.info("Training history saved to %s", csv_path)

    def _config_to_dict(self) -> dict:
        """Convert config to dict for serialization."""
        import dataclasses
        def to_dict(obj):
            if dataclasses.is_dataclass(obj):
                return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
            elif isinstance(obj, list):
                return [to_dict(i) for i in obj]
            return obj
        return to_dict(self.config)
