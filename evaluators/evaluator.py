"""
Full evaluation pipeline for WindGapGAN.

Orchestrates model evaluation on test data, classical baselines
comparison, metric computation, and result export.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from evaluators.metrics import compute_all_metrics, compute_gap_metrics

logger = logging.getLogger(__name__)


class Evaluator:
    """
    Comprehensive evaluation pipeline.

    Runs the model on test data, computes all metric groups,
    compares against classical baselines, and exports results.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        norm_stats: Optional[dict[str, Any]] = None,
        output_dir: str | Path = "outputs",
        regimes: Optional[list[list[Optional[float]]]] = None,
    ) -> None:
        self.model = model
        self.device = device
        self.norm_stats = norm_stats
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.regimes = regimes

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        run_baselines: bool = True,
    ) -> dict[str, Any]:
        """
        Run full evaluation on a dataset.

        Args:
            dataloader: DataLoader for evaluation data.
            run_baselines: Whether to also run classical baselines.

        Returns:
            Dict with model metrics and (optionally) baseline metrics.
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_masks = []

        for batch in dataloader:
            inputs = batch["input"].to(self.device)   # (B, T, 2, H, W)
            targets = batch["target"]                  # (B, T, 1, H, W)
            masks = batch["mask"]                      # (B, T, 1, H, W)

            preds = self.model(inputs).cpu()           # (B, T, 1, H, W)

            all_preds.append(preds.numpy())
            all_targets.append(targets.numpy())
            all_masks.append(masks.numpy())

        # Concatenate all batches
        preds = np.concatenate(all_preds, axis=0)     # (N, T, 1, H, W)
        targets = np.concatenate(all_targets, axis=0)
        masks = np.concatenate(all_masks, axis=0)

        # Denormalize if needed
        if self.norm_stats and self.norm_stats["method"] != "none":
            from datasets.nc_dataset import denormalize
            preds = denormalize(preds, self.norm_stats)
            targets = denormalize(targets, self.norm_stats)

        # Flatten spatial dimensions for metric computation
        preds_flat = preds.reshape(-1)
        targets_flat = targets.reshape(-1)
        masks_flat = masks.reshape(-1)

        # Compute all metrics
        results = {
            "model": compute_all_metrics(preds_flat, targets_flat, masks_flat, self.regimes),
        }

        # Run classical baselines
        if run_baselines:
            from models.classical_baselines import CLASSICAL_BASELINES

            # Reshape for baseline format: (N*T, H, W)
            N, T, C, H, W = preds.shape
            target_3d = targets.reshape(N * T, H, W)
            mask_3d = masks.reshape(N * T, H, W)
            input_3d = (target_3d * mask_3d)  # Masked input

            for name, baseline_cls in CLASSICAL_BASELINES.items():
                logger.info("Running baseline: %s", baseline_cls.name)
                baseline_pred = baseline_cls.fill(input_3d, mask_3d)
                baseline_metrics = compute_gap_metrics(
                    baseline_pred.reshape(-1),
                    target_3d.reshape(-1),
                    mask_3d.reshape(-1),
                )
                results[name] = {"gap": baseline_metrics}

        # Save results
        self._save_results(results)

        return results

    def _save_results(self, results: dict[str, Any]) -> None:
        """Save evaluation results to CSV and summary markdown."""
        # CSV: flatten metrics for tabular comparison
        rows = []
        for method, metrics_dict in results.items():
            gap_metrics = metrics_dict.get("gap", {})
            row = {"method": method}
            row.update(gap_metrics)
            rows.append(row)

        df = pd.DataFrame(rows)
        csv_path = self.output_dir / "evaluation_results.csv"
        df.to_csv(csv_path, index=False, float_format="%.6f")
        logger.info("Evaluation results saved to %s", csv_path)

        # Summary markdown
        summary_path = self.output_dir / "experiment_summary.md"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("# Experiment Summary\n\n")
            f.write("## Gap Reconstruction Metrics\n\n")
            f.write(df.to_markdown(index=False, floatfmt=".6f"))
            f.write("\n\n")

            # Stratified metrics (model only)
            if "model" in results and "stratified" in results["model"]:
                f.write("## Stratified Metrics (Model)\n\n")
                strat = results["model"]["stratified"]
                for regime, metrics in strat.items():
                    f.write(f"### Regime: {regime}\n")
                    for k, v in metrics.items():
                        f.write(f"- {k}: {v}\n")
                    f.write("\n")

        logger.info("Experiment summary saved to %s", summary_path)
