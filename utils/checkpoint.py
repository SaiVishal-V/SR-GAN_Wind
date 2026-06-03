"""
Checkpoint management for WindGapGAN.

Handles saving and loading of model checkpoints, including:
- Latest checkpoint
- Best checkpoint per metric (RMSE, MAE, Correlation)
- Periodic checkpoints
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with support for best-metric tracking.

    Tracks:
        - latest: Always overwrites with the most recent state
        - best_rmse: Saves when RMSE improves (lower is better)
        - best_mae: Saves when MAE improves (lower is better)
        - best_corr: Saves when Correlation improves (higher is better)
        - periodic: Saves every N epochs
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_best_rmse: bool = True,
        save_best_mae: bool = True,
        save_best_corr: bool = True,
        save_every: int = 50,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.save_best_rmse = save_best_rmse
        self.save_best_mae = save_best_mae
        self.save_best_corr = save_best_corr
        self.save_every = save_every

        # Track best metrics
        self.best_rmse: float = float("inf")
        self.best_mae: float = float("inf")
        self.best_corr: float = float("-inf")

    def save(
        self,
        state: dict[str, Any],
        epoch: int,
        metrics: dict[str, float],
    ) -> list[str]:
        """
        Save checkpoints based on current metrics.

        Args:
            state: Dict containing model_state_dict, optimizer_state_dict, etc.
            epoch: Current epoch number.
            metrics: Dict with keys like 'rmse_gap', 'mae_gap', 'corr_gap'.

        Returns:
            List of checkpoint paths that were saved.
        """
        saved = []

        # Add epoch to state
        state["epoch"] = epoch
        state["metrics"] = metrics

        # Always save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(state, latest_path)
        saved.append(str(latest_path))

        # Best RMSE (lower is better)
        rmse = metrics.get("rmse_gap", float("inf"))
        if self.save_best_rmse and rmse < self.best_rmse:
            self.best_rmse = rmse
            best_path = self.checkpoint_dir / "best_rmse.pt"
            torch.save(state, best_path)
            saved.append(str(best_path))
            logger.info("New best RMSE: %.6f (epoch %d)", rmse, epoch)

        # Best MAE (lower is better)
        mae = metrics.get("mae_gap", float("inf"))
        if self.save_best_mae and mae < self.best_mae:
            self.best_mae = mae
            best_path = self.checkpoint_dir / "best_mae.pt"
            torch.save(state, best_path)
            saved.append(str(best_path))
            logger.info("New best MAE: %.6f (epoch %d)", mae, epoch)

        # Best Correlation (higher is better)
        corr = metrics.get("corr_gap", float("-inf"))
        if self.save_best_corr and corr > self.best_corr:
            self.best_corr = corr
            best_path = self.checkpoint_dir / "best_corr.pt"
            torch.save(state, best_path)
            saved.append(str(best_path))
            logger.info("New best Correlation: %.6f (epoch %d)", corr, epoch)

        # Periodic save
        if self.save_every > 0 and (epoch + 1) % self.save_every == 0:
            periodic_path = self.checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
            torch.save(state, periodic_path)
            saved.append(str(periodic_path))

        return saved

    @staticmethod
    def load(
        checkpoint_path: str | Path,
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[torch.device] = None,
    ) -> dict[str, Any]:
        """
        Load a checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file.
            model: Model to load state dict into.
            optimizer: Optional optimizer to load state dict into.
            device: Device to map tensors to.

        Returns:
            Full checkpoint dict (epoch, metrics, etc.)
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        map_location = device if device else "cpu"
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)

        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Model state loaded from %s", checkpoint_path)

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            logger.info("Optimizer state loaded from %s", checkpoint_path)

        return checkpoint
