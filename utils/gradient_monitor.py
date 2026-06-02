"""
Gradient monitoring utility for training diagnostics.

Tracks gradient statistics per-layer and per-epoch to detect:
    - Vanishing gradients (norm < threshold)
    - Exploding gradients (norm > threshold)
    - Gradient imbalance between layers

Logs to TensorBoard and/or returns summary dict for experiment reports.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List


class GradientMonitor:
    """
    Monitors gradient statistics during training.

    Usage:
        monitor = GradientMonitor(generator)

        for batch in train_loader:
            loss.backward()
            stats = monitor.compute_stats(generator)
            monitor.log(writer, epoch)
            optimizer.step()

    Args:
        vanishing_threshold: Gradient norm below this is flagged as vanishing.
        exploding_threshold: Gradient norm above this is flagged as exploding.
    """

    def __init__(
        self,
        vanishing_threshold: float = 1e-7,
        exploding_threshold: float = 100.0,
    ):
        self.vanishing_threshold = vanishing_threshold
        self.exploding_threshold = exploding_threshold
        self.history: List[Dict[str, float]] = []

    def compute_stats(self, model: nn.Module) -> Dict[str, float]:
        """
        Compute gradient statistics for the model.

        Args:
            model: Model with computed gradients (after backward()).

        Returns:
            Dictionary of gradient statistics.
        """
        norms = []
        layer_norms = {}

        for name, param in model.named_parameters():
            if param.grad is not None:
                norm = param.grad.data.norm(2).item()
                norms.append(norm)

                # Group by top-level module
                group = name.split(".")[0]
                if group not in layer_norms:
                    layer_norms[group] = []
                layer_norms[group].append(norm)

        if not norms:
            return {"grad_norm_mean": 0.0, "grad_norm_max": 0.0,
                    "grad_norm_min": 0.0, "grad_variance": 0.0,
                    "vanishing_count": 0, "exploding_count": 0}

        stats = {
            "grad_norm_mean": float(np.mean(norms)),
            "grad_norm_max": float(np.max(norms)),
            "grad_norm_min": float(np.min(norms)),
            "grad_variance": float(np.var(norms)),
            "vanishing_count": sum(1 for n in norms if n < self.vanishing_threshold),
            "exploding_count": sum(1 for n in norms if n > self.exploding_threshold),
            "total_params_with_grad": len(norms),
        }

        # Per-layer group stats
        for group, gnorms in layer_norms.items():
            stats[f"grad_norm_{group}_mean"] = float(np.mean(gnorms))
            stats[f"grad_norm_{group}_max"] = float(np.max(gnorms))

        self.history.append(stats)
        return stats

    def log_to_tensorboard(
        self,
        writer,
        epoch: int,
        prefix: str = "gradients",
        model: Optional[nn.Module] = None,
    ) -> None:
        """
        Log gradient statistics to TensorBoard.

        Args:
            writer: TensorBoard SummaryWriter.
            epoch: Current epoch number.
            prefix: Metric prefix.
            model: Optional model to log parameter histograms.
        """
        if not self.history:
            return

        stats = self.history[-1]

        writer.add_scalar(f"{prefix}/norm_mean", stats["grad_norm_mean"], epoch)
        writer.add_scalar(f"{prefix}/norm_max", stats["grad_norm_max"], epoch)
        writer.add_scalar(f"{prefix}/norm_min", stats["grad_norm_min"], epoch)
        writer.add_scalar(f"{prefix}/variance", stats["grad_variance"], epoch)
        writer.add_scalar(f"{prefix}/vanishing_count", stats["vanishing_count"], epoch)
        writer.add_scalar(f"{prefix}/exploding_count", stats["exploding_count"], epoch)

        # Log gradient histograms (expensive, do periodically)
        if model is not None and epoch % 10 == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(
                        f"{prefix}/grad_{name}",
                        param.grad.data.cpu(),
                        epoch,
                    )

    def check_alerts(self) -> List[str]:
        """
        Check for gradient health alerts.

        Returns:
            List of alert messages (empty if healthy).
        """
        if not self.history:
            return []

        stats = self.history[-1]
        alerts = []

        if stats["vanishing_count"] > 0:
            alerts.append(
                f"WARNING: {stats['vanishing_count']}/{stats['total_params_with_grad']} "
                f"parameters have vanishing gradients (norm < {self.vanishing_threshold})"
            )

        if stats["exploding_count"] > 0:
            alerts.append(
                f"WARNING: {stats['exploding_count']}/{stats['total_params_with_grad']} "
                f"parameters have exploding gradients (norm > {self.exploding_threshold})"
            )

        return alerts

    def get_summary(self) -> Dict[str, float]:
        """
        Get summary statistics across all recorded epochs.

        Returns:
            Dictionary with overall gradient health summary.
        """
        if not self.history:
            return {}

        means = [h["grad_norm_mean"] for h in self.history]
        maxes = [h["grad_norm_max"] for h in self.history]
        variances = [h["grad_variance"] for h in self.history]

        return {
            "avg_grad_norm": float(np.mean(means)),
            "max_grad_norm_ever": float(np.max(maxes)),
            "avg_grad_variance": float(np.mean(variances)),
            "total_vanishing_alerts": sum(
                h["vanishing_count"] > 0 for h in self.history
            ),
            "total_exploding_alerts": sum(
                h["exploding_count"] > 0 for h in self.history
            ),
            "epochs_tracked": len(self.history),
        }
