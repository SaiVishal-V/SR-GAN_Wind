"""
Checkpoint forensics utility.

Loads and evaluates any available checkpoints to determine:
  - Performance trajectory during training
  - Whether performance improved, stagnated, or degraded
  - Best vs last checkpoint comparison

Usage:
    python analysis/checkpoint_analysis.py --config configs/config.yaml --checkpoint-dir checkpoints/
    python analysis/checkpoint_analysis.py --config configs/config.yaml --checkpoint checkpoints/best_rmse_generator.pth
"""

import argparse
import os
import sys
import json
import glob

import numpy as np
import torch
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.seed import set_seed
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_all_metrics, NORM_MEAN, NORM_STD
from datasets.wind_dataset import WindSRDataset, get_temporal_split
from models.generator import SRResNet


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_checkpoint(generator, test_dataset, device, max_scenes=20):
    """Evaluate a generator on test scenes. Returns dict of mean metrics."""
    generator.eval()
    all_metrics = {}

    n = min(len(test_dataset), max_scenes)
    for i in tqdm(range(n), desc="  Evaluating", leave=False):
        sample = test_dataset[i]
        lr = sample["lr"].unsqueeze(0).to(device)
        hr = sample["hr"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)

        sr = generator(lr)
        metrics = compute_all_metrics(sr, hr, mask)

        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in all_metrics.items()}


def analyze_checkpoints(config, checkpoint_paths, device, test_dataset):
    """Analyze multiple checkpoints and generate forensics report."""
    gen_cfg = config["model"]["generator"]
    results = {}

    for ckpt_path in checkpoint_paths:
        name = os.path.basename(ckpt_path)
        print(f"\n  Loading: {name}")

        try:
            generator = SRResNet(
                in_channels=config["model"]["in_channels"],
                num_features=gen_cfg["num_features"],
                num_residual_blocks=gen_cfg["num_residual_blocks"],
                scale_factor=config["model"]["scale_factor"],
            ).to(device)

            ckpt = load_checkpoint(ckpt_path, device)
            generator.load_state_dict(ckpt["generator_state_dict"])

            # Extract metadata
            meta = {
                "epoch": ckpt.get("epoch", "unknown"),
                "stage": ckpt.get("stage", "unknown"),
                "best_metrics": ckpt.get("best_metrics", {}),
            }

            # Evaluate
            metrics = evaluate_checkpoint(generator, test_dataset, device)

            results[name] = {
                "metadata": meta,
                "metrics": metrics,
            }

            print(f"    Epoch: {meta['epoch']}, Stage: {meta['stage']}")
            print(f"    RMSE:  {metrics.get('rmse', {}).get('mean', 'N/A'):.6f}")
            print(f"    SSIM:  {metrics.get('ssim', {}).get('mean', 'N/A'):.6f}")
            print(f"    PSNR:  {metrics.get('psnr', {}).get('mean', 'N/A'):.6f}")

        except Exception as e:
            print(f"    ERROR: {e}")
            results[name] = {"error": str(e)}

    return results


def generate_report(results, output_path):
    """Generate markdown forensics report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w") as f:
        f.write("# Checkpoint Forensics Report\n\n")

        if not results:
            f.write("**No checkpoints found or evaluated.**\n\n")
            f.write("This report will be populated after the first training run completes.\n")
            return

        # Summary table
        f.write("## Performance Summary\n\n")
        f.write("| Checkpoint | Epoch | Stage | RMSE | MAE | SSIM | PSNR | Grad RMSE |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")

        for name, data in results.items():
            if "error" in data:
                f.write(f"| {name} | ERROR | — | — | — | — | — | — |\n")
                continue

            meta = data["metadata"]
            m = data["metrics"]
            f.write(
                f"| {name} "
                f"| {meta.get('epoch', '?')} "
                f"| {meta.get('stage', '?')} "
                f"| {m.get('rmse', {}).get('mean', 0):.6f} "
                f"| {m.get('mae', {}).get('mean', 0):.6f} "
                f"| {m.get('ssim', {}).get('mean', 0):.4f} "
                f"| {m.get('psnr', {}).get('mean', 0):.2f} "
                f"| {m.get('gradient_rmse', {}).get('mean', 0):.6f} |\n"
            )

        # Trajectory analysis
        f.write("\n## Trajectory Analysis\n\n")
        epochs = []
        rmses = []
        for name, data in results.items():
            if "error" not in data:
                ep = data["metadata"].get("epoch", -1)
                rmse = data["metrics"].get("rmse", {}).get("mean", None)
                if isinstance(ep, int) and rmse is not None:
                    epochs.append(ep)
                    rmses.append(rmse)

        if len(epochs) >= 2:
            sorted_pairs = sorted(zip(epochs, rmses))
            first_rmse = sorted_pairs[0][1]
            last_rmse = sorted_pairs[-1][1]
            best_rmse = min(rmses)

            if last_rmse < first_rmse:
                f.write("**Performance IMPROVED** during training.\n\n")
            elif last_rmse > first_rmse * 1.05:
                f.write("**Performance DEGRADED** during training (possible overfitting).\n\n")
            else:
                f.write("**Performance STAGNATED** during training.\n\n")

            f.write(f"- First checkpoint RMSE: {first_rmse:.6f}\n")
            f.write(f"- Last checkpoint RMSE:  {last_rmse:.6f}\n")
            f.write(f"- Best checkpoint RMSE:  {best_rmse:.6f}\n")
        else:
            f.write("Insufficient checkpoints for trajectory analysis.\n")

    print(f"\n  Report saved: {output_path}")

    # Also save raw JSON
    json_path = output_path.replace(".md", ".json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  JSON saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Checkpoint Forensics")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Single checkpoint to analyze")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Directory of checkpoints to analyze")
    parser.add_argument("--max-scenes", type=int, default=20,
                        help="Max test scenes per checkpoint")
    parser.add_argument("--colab", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"], deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find checkpoints
    checkpoint_paths = []
    if args.checkpoint:
        checkpoint_paths = [args.checkpoint]
    elif args.checkpoint_dir:
        checkpoint_paths = sorted(glob.glob(os.path.join(args.checkpoint_dir, "*.pth")))
    else:
        # Auto-detect
        default_dir = config["checkpoint"]["save_dir"]
        if args.colab:
            default_dir = os.path.join("/content/drive/MyDrive/SR-GAN_Wind", default_dir)
        if os.path.isdir(default_dir):
            checkpoint_paths = sorted(glob.glob(os.path.join(default_dir, "*.pth")))

    if not checkpoint_paths:
        print("No checkpoints found. Generating empty report.")
        generate_report({}, "reports/checkpoint_forensics.md")
        return

    print(f"Found {len(checkpoint_paths)} checkpoint(s)")

    # Create test dataset
    data_path = config["data"]["colab_path"] if args.colab else config["data"]["local_path"]
    import netCDF4
    ds = netCDF4.Dataset(data_path, "r")
    n_time = ds.variables[config["data"]["lr_variable"]].shape[0]
    ds.close()

    _, _, test_idx = get_temporal_split(
        n_time, config["data"]["train_ratio"],
        config["data"]["val_ratio"], config["data"]["test_ratio"],
    )

    test_dataset = WindSRDataset(
        nc_path=data_path,
        lr_variable=config["data"]["lr_variable"],
        hr_variable=config["data"]["hr_variable"],
        mask_variable=config["data"]["mask_variable"],
        quality_variable=config["data"]["quality_variable"],
        time_indices=test_idx,
        mode="full",
        cache=config["data"]["cache_dataset"],
    )

    # Analyze
    results = analyze_checkpoints(config, checkpoint_paths, device, test_dataset)

    # Report
    generate_report(results, "reports/checkpoint_forensics.md")


if __name__ == "__main__":
    main()
