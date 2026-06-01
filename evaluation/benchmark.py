"""
Benchmark script — automated comparison of baselines (review #8 & #14).

Compares:
    1. Bicubic Interpolation (F.interpolate, mode='bicubic')
    2. SRResNet (Generator-only, pretrained)
    3. SR-GAN (Generator after GAN fine-tuning)

Generates:
    - benchmark_report.csv
    - benchmark_report.md
    - Comparison plots for each baseline

Usage:
    python benchmark.py --config configs/config.yaml \
        --srresnet checkpoints/pretrain_best.pth \
        --srgan checkpoints/best_rmse_generator.pth
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.seed import set_seed
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_all_metrics, NORM_MEAN, NORM_STD
from utils.plotting import plot_comparison
from datasets.wind_dataset import WindSRDataset, get_temporal_split
from models.generator import SRResNet


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_bicubic(test_dataset, device, scale_factor=4) -> dict:
    """Evaluate bicubic interpolation baseline."""
    print("\n  Evaluating: Bicubic Interpolation")
    all_metrics = {}

    for i in tqdm(range(len(test_dataset)), desc="Bicubic"):
        sample = test_dataset[i]
        lr = sample["lr"].unsqueeze(0).to(device)
        hr = sample["hr"].unsqueeze(0).to(device)
        mask = sample["mask"].unsqueeze(0).to(device)

        # Bicubic upsampling
        sr = F.interpolate(
            lr, scale_factor=scale_factor, mode="bicubic", align_corners=False
        )

        metrics = compute_all_metrics(sr, hr, mask)
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


@torch.no_grad()
def evaluate_generator(
    generator, test_dataset, device, model_name="SRResNet"
) -> dict:
    """Evaluate a generator model."""
    print(f"\n  Evaluating: {model_name}")
    generator.eval()
    all_metrics = {}

    for i in tqdm(range(len(test_dataset)), desc=model_name):
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

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


def generate_report(results: dict, output_dir: str) -> None:
    """Generate benchmark_report.csv and benchmark_report.md."""
    os.makedirs(output_dir, exist_ok=True)

    models = list(results.keys())
    if not models:
        return

    metrics = sorted(results[models[0]].keys())

    # CSV
    csv_path = os.path.join(output_dir, "benchmark_report.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model"] + metrics)
        for model in models:
            row = [model] + [f"{results[model].get(m, 0.0):.6f}" for m in metrics]
            writer.writerow(row)
    print(f"\n  CSV report: {csv_path}")

    # Markdown
    md_path = os.path.join(output_dir, "benchmark_report.md")
    with open(md_path, "w") as f:
        f.write("# SR-GAN Benchmark Report\n\n")

        # Key metrics table
        key_metrics = ["rmse", "mae", "psnr", "ssim", "gradient_rmse",
                       "rmse_mps", "mae_mps", "bias_mps", "correlation"]
        available = [m for m in key_metrics if m in metrics]

        f.write("## Results Summary\n\n")
        f.write("| Model | " + " | ".join(available) + " |\n")
        f.write("|" + "---|" * (len(available) + 1) + "\n")

        for model in models:
            vals = [f"{results[model].get(m, 0.0):.4f}" for m in available]
            f.write(f"| {model} | " + " | ".join(vals) + " |\n")

        f.write("\n## All Metrics\n\n")
        f.write("| Model | " + " | ".join(metrics) + " |\n")
        f.write("|" + "---|" * (len(metrics) + 1) + "\n")

        for model in models:
            vals = [f"{results[model].get(m, 0.0):.6f}" for m in metrics]
            f.write(f"| {model} | " + " | ".join(vals) + " |\n")

    print(f"  Markdown report: {md_path}")


def main():
    parser = argparse.ArgumentParser(description="SR-GAN Benchmark")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--srresnet", type=str, default=None,
                        help="Path to pretrained SRResNet checkpoint.")
    parser.add_argument("--srgan", type=str, default=None,
                        help="Path to GAN-finetuned generator checkpoint.")
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs/benchmark")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["seed"], deterministic=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    data_path = config["data"]["colab_path"] if args.colab else config["data"]["local_path"]

    import netCDF4
    ds = netCDF4.Dataset(data_path, "r")
    n_time = ds.variables[config["data"]["lr_variable"]].shape[0]
    ds.close()

    _, _, test_idx = get_temporal_split(
        n_time,
        config["data"]["train_ratio"],
        config["data"]["val_ratio"],
        config["data"]["test_ratio"],
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

    results = {}

    # --- Baseline 1: Bicubic ---
    results["Bicubic"] = evaluate_bicubic(
        test_dataset, device, config["model"]["scale_factor"]
    )

    # --- Baseline 2: SRResNet ---
    gen_cfg = config["model"]["generator"]

    if args.srresnet:
        generator = SRResNet(
            in_channels=config["model"]["in_channels"],
            num_features=gen_cfg["num_features"],
            num_residual_blocks=gen_cfg["num_residual_blocks"],
            scale_factor=config["model"]["scale_factor"],
        ).to(device)

        ckpt = load_checkpoint(args.srresnet, device)
        generator.load_state_dict(ckpt["generator_state_dict"])
        results["SRResNet"] = evaluate_generator(
            generator, test_dataset, device, "SRResNet"
        )

    # --- Baseline 3: SR-GAN ---
    if args.srgan:
        generator = SRResNet(
            in_channels=config["model"]["in_channels"],
            num_features=gen_cfg["num_features"],
            num_residual_blocks=gen_cfg["num_residual_blocks"],
            scale_factor=config["model"]["scale_factor"],
        ).to(device)

        ckpt = load_checkpoint(args.srgan, device)
        generator.load_state_dict(ckpt["generator_state_dict"])
        results["SRGAN"] = evaluate_generator(
            generator, test_dataset, device, "SRGAN"
        )

    # Generate reports
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for model, metrics in results.items():
        print(f"\n  {model}:")
        for k, v in sorted(metrics.items()):
            print(f"    {k:20s}: {v:.6f}")

    generate_report(results, args.output_dir)


if __name__ == "__main__":
    main()
