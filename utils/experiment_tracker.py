"""
Experiment tracker for reproducibility and ablation studies.

Saves per-run:
    - config_used.yaml (exact configuration used)
    - experiment_summary.md (results, metadata, gradient health)

Optional W&B integration with graceful fallback to TensorBoard-only.
"""

import os
import yaml
import json
import copy
from datetime import datetime, timezone
from typing import Dict, Optional, Any


class ExperimentTracker:
    """
    Tracks experiment configuration, metrics, and produces reproducibility artifacts.

    Usage:
        tracker = ExperimentTracker(
            experiment_name="E2_ema",
            config=config,
            output_dir="experiments/E2_ema",
        )
        tracker.save_config()

        # ... training ...

        tracker.log_metrics(epoch, {"rmse": 0.05, "ssim": 0.95})
        tracker.save_summary(best_metrics, gradient_summary)
    """

    def __init__(
        self,
        experiment_name: str,
        config: Dict,
        output_dir: str,
        wandb_enabled: bool = False,
        wandb_project: str = "srgan-wind",
    ):
        self.experiment_name = experiment_name
        self.config = copy.deepcopy(config)
        self.output_dir = output_dir
        self.start_time = datetime.now(timezone.utc)
        self.metrics_history: list = []

        os.makedirs(output_dir, exist_ok=True)

        # W&B integration (optional)
        self.wandb_run = None
        if wandb_enabled:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=experiment_name,
                    config=config,
                    dir=output_dir,
                )
                print(f"  W&B initialized: {wandb_project}/{experiment_name}")
            except ImportError:
                print("  W&B not installed. Falling back to TensorBoard-only.")
            except Exception as e:
                print(f"  W&B init failed: {e}. Falling back to TensorBoard-only.")

    def save_config(self) -> str:
        """Save the exact configuration used for this experiment."""
        path = os.path.join(self.output_dir, "config_used.yaml")
        with open(path, "w") as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        return path

    def log_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        stage: str = "train",
    ) -> None:
        """
        Log metrics for an epoch.

        Args:
            epoch: Current epoch number.
            metrics: Dictionary of metric values.
            stage: 'train', 'pretrain', or 'gan'.
        """
        entry = {"epoch": epoch, "stage": stage}
        entry.update(metrics)
        self.metrics_history.append(entry)

        # Log to W&B if available
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.log({f"{stage}/{k}": v for k, v in metrics.items()}, step=epoch)
            except Exception:
                pass

    def save_summary(
        self,
        best_metrics: Optional[Dict[str, float]] = None,
        gradient_summary: Optional[Dict[str, float]] = None,
        additional_notes: str = "",
    ) -> str:
        """
        Generate experiment_summary.md with full results.

        Args:
            best_metrics: Best metric values achieved.
            gradient_summary: Gradient health summary from GradientMonitor.
            additional_notes: Free-form notes about the experiment.

        Returns:
            Path to the saved summary file.
        """
        end_time = datetime.now(timezone.utc)
        duration = end_time - self.start_time

        path = os.path.join(self.output_dir, "experiment_summary.md")
        with open(path, "w") as f:
            f.write(f"# Experiment Summary: {self.experiment_name}\n\n")

            # Metadata
            f.write("## Metadata\n\n")
            f.write(f"| Property | Value |\n")
            f.write(f"|----------|-------|\n")
            f.write(f"| Experiment | {self.experiment_name} |\n")
            f.write(f"| Start Time | {self.start_time.isoformat()} |\n")
            f.write(f"| End Time | {end_time.isoformat()} |\n")
            f.write(f"| Duration | {duration} |\n")
            f.write(f"| Seed | {self.config.get('seed', 'N/A')} |\n")

            # Key config
            opt = self.config.get("optimization", {})
            loss = self.config.get("loss", {})
            f.write(f"| Scheduler | {opt.get('scheduler_type', 'cosine')} |\n")
            f.write(f"| EMA | {opt.get('ema_enabled', False)} |\n")
            f.write(f"| Pixel Loss Type | {loss.get('pixel_loss_type', 'l1')} |\n")
            f.write(f"| Laplacian Weight | {loss.get('laplacian_weight', 0.0)} |\n")
            f.write(f"| Spectral Weight | {loss.get('spectral_weight', 0.0)} |\n")
            f.write(f"| Adversarial Weight | {loss.get('adversarial_weight', 0.001)} |\n")

            # Best metrics
            if best_metrics:
                f.write("\n## Best Metrics\n\n")
                f.write("| Metric | Value |\n")
                f.write("|--------|-------|\n")
                for k, v in sorted(best_metrics.items()):
                    f.write(f"| {k} | {v:.6f} |\n")

            # Gradient health
            if gradient_summary:
                f.write("\n## Gradient Health\n\n")
                f.write("| Property | Value |\n")
                f.write("|----------|-------|\n")
                for k, v in sorted(gradient_summary.items()):
                    f.write(f"| {k} | {v} |\n")

            # Notes
            if additional_notes:
                f.write(f"\n## Notes\n\n{additional_notes}\n")

        # Also save metrics history as JSON
        json_path = os.path.join(self.output_dir, "metrics_history.json")
        with open(json_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        return path

    def close(self) -> None:
        """Finish experiment tracking."""
        if self.wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
