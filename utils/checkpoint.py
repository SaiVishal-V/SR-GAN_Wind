"""
Checkpoint utilities for saving and loading model states.
Supports best-metric checkpoints (RMSE, SSIM, PSNR) and resume training.
"""

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state: Dict[str, Any],
    filepath: str,
) -> None:
    """
    Save a training checkpoint.

    Args:
        state: Dictionary containing model_state, optimizer_state,
               scheduler_state, epoch, best_metrics, etc.
        filepath: Path to save the checkpoint.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)


def load_checkpoint(
    filepath: str,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load a training checkpoint.

    Args:
        filepath: Path to the checkpoint file.
        device: Device to map tensors to.

    Returns:
        Checkpoint dictionary.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")

    checkpoint = torch.load(filepath, map_location=device, weights_only=False)
    return checkpoint


def build_checkpoint_state(
    epoch: int,
    generator: torch.nn.Module,
    discriminator: Optional[torch.nn.Module],
    optimizer_g: torch.optim.Optimizer,
    optimizer_d: Optional[torch.optim.Optimizer],
    scheduler_g: Optional[Any],
    scheduler_d: Optional[Any],
    best_metrics: Dict[str, float],
    stage: str,
) -> Dict[str, Any]:
    """
    Build a checkpoint state dictionary.

    Args:
        epoch: Current epoch number.
        generator: Generator model.
        discriminator: Discriminator model (None during pretraining).
        optimizer_g: Generator optimizer.
        optimizer_d: Discriminator optimizer (None during pretraining).
        scheduler_g: Generator LR scheduler.
        scheduler_d: Discriminator LR scheduler.
        best_metrics: Dictionary of best metric values.
        stage: Current training stage ('pretrain' or 'gan').

    Returns:
        Checkpoint state dictionary.
    """
    state = {
        "epoch": epoch,
        "stage": stage,
        "generator_state_dict": generator.state_dict(),
        "optimizer_g_state_dict": optimizer_g.state_dict(),
        "best_metrics": best_metrics,
    }

    if discriminator is not None:
        state["discriminator_state_dict"] = discriminator.state_dict()

    if optimizer_d is not None:
        state["optimizer_d_state_dict"] = optimizer_d.state_dict()

    if scheduler_g is not None:
        state["scheduler_g_state_dict"] = scheduler_g.state_dict()

    if scheduler_d is not None:
        state["scheduler_d_state_dict"] = scheduler_d.state_dict()

    return state


class BestMetricTracker:
    """
    Tracks best metric values and saves checkpoints when improved.
    Supports RMSE (lower is better), SSIM (higher is better), PSNR (higher is better).
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.best = {
            "rmse": float("inf"),
            "ssim": 0.0,
            "psnr": 0.0,
        }

    def update(
        self, metric_name: str, value: float, state: Dict[str, Any]
    ) -> bool:
        """
        Update best metric and save checkpoint if improved.

        Args:
            metric_name: One of 'rmse', 'ssim', 'psnr'.
            value: Current metric value.
            state: Checkpoint state to save.

        Returns:
            True if the metric improved and checkpoint was saved.
        """
        improved = False

        if metric_name == "rmse" and value < self.best["rmse"]:
            self.best["rmse"] = value
            improved = True
        elif metric_name == "ssim" and value > self.best["ssim"]:
            self.best["ssim"] = value
            improved = True
        elif metric_name == "psnr" and value > self.best["psnr"]:
            self.best["psnr"] = value
            improved = True

        if improved:
            path = os.path.join(self.save_dir, f"best_{metric_name}_generator.pth")
            save_checkpoint(state, path)

        return improved

    def get_best(self) -> Dict[str, float]:
        """Return dictionary of best metric values."""
        return dict(self.best)
