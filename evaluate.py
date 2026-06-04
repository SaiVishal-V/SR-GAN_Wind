"""
WindGapGAN — Main Evaluation Entry Point.

Usage:
    python evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_rmse.pt --data.nc_path /path/to/data.nc
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import parse_args_and_load_config
from utils.logging_utils import setup_logging
from utils.seed import set_seed


def main() -> None:
    """Main evaluation entry point."""
    # Parse args — we need --checkpoint in addition to standard config args
    parser = argparse.ArgumentParser(
        description="WindGapGAN — Evaluation",
        add_help=False,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    known_args, remaining = parser.parse_known_args()

    # Re-inject remaining args for config parsing
    sys.argv = [sys.argv[0]] + remaining
    config = parse_args_and_load_config()

    setup_logging(log_dir=config.logging.log_dir)
    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("WindGapGAN — Evaluation")
    logger.info("=" * 60)

    set_seed(config.seed, deterministic=True)
    device = config.resolve_device()

    # Build test dataset
    from datasets.nc_dataset import build_datasets

    _, _, test_ds, metadata = build_datasets(
        nc_path=str(config.data.nc_path),
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

    from torch.utils.data import DataLoader

    test_loader = DataLoader(
        test_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
    )

    # Load model
    from models.unet import MaskedUNet
    from utils.checkpoint import CheckpointManager

    model_name = config.model.name.lower()
    if model_name in ("unet", "gan"):
        model = MaskedUNet(
            in_channels=config.model.in_channels,
            out_channels=config.model.out_channels,
            base_features=config.model.base_features,
            depth=config.model.depth,
            dropout=0.0,  # No dropout at eval
            use_batch_norm=config.model.use_batch_norm,
            use_attention=config.model.use_attention,
            use_tanh=config.model.use_tanh,
            hard_merge=True,  # Always hard merge at evaluation
        )
    else:
        raise NotImplementedError(f"Model '{model_name}' not yet implemented for evaluation.")

    checkpoint = CheckpointManager.load(known_args.checkpoint, model, device=device)
    model.to(device)
    logger.info("Loaded checkpoint from epoch %d", checkpoint.get("epoch", -1))

    # Run evaluation
    from evaluators.evaluator import Evaluator

    evaluator = Evaluator(
        model=model,
        device=device,
        norm_stats=metadata.get("norm_stats"),
        output_dir=config.output.output_dir,
        regimes=config.evaluation.wind_speed_regimes,
    )

    results = evaluator.evaluate(test_loader, run_baselines=config.evaluation.run_classical_baselines)

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)

    for method, metrics in results.items():
        gap_metrics = metrics.get("gap", {})
        logger.info(
            "%s │ RMSE=%.4f │ MAE=%.4f │ Bias=%.4f │ Corr=%.4f",
            method.ljust(25),
            gap_metrics.get("rmse_gap", float("nan")),
            gap_metrics.get("mae_gap", float("nan")),
            gap_metrics.get("bias_gap", float("nan")),
            gap_metrics.get("corr_gap", float("nan")),
        )

    logger.info("Results saved to %s", config.output.output_dir)


if __name__ == "__main__":
    main()
