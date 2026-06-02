"""
Monthly evaluation — compute per-month metrics on test set with uncertainty.

Outputs:
    - monthly_metrics.csv
    - monthly_metrics.md (with mean ± std + 95% CI)

Usage:
    python evaluation/monthly_evaluation.py --config configs/config.yaml --checkpoint checkpoints/best_rmse_generator.pth
"""
import argparse, os, sys, json, csv
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.seed import set_seed
from utils.metrics import compute_all_metrics
from datasets.wind_dataset import WindSRDataset, get_temporal_split
from models.generator import SRResNet
from utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    set_seed(config["seed"], deterministic=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    gen_cfg = config["model"]["generator"]
    generator = SRResNet(
        in_channels=config["model"]["in_channels"],
        num_features=gen_cfg["num_features"],
        num_residual_blocks=gen_cfg["num_residual_blocks"],
        scale_factor=config["model"]["scale_factor"],
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()

    # Create test dataset with time info
    data_path = config["data"]["colab_path"] if args.colab else config["data"]["local_path"]
    import netCDF4
    ds_nc = netCDF4.Dataset(data_path, "r")
    n_time = ds_nc.variables[config["data"]["lr_variable"]].shape[0]
    ds_nc.close()

    _, _, test_idx = get_temporal_split(
        n_time, config["data"]["train_ratio"],
        config["data"]["val_ratio"], config["data"]["test_ratio"],
    )

    test_dataset = WindSRDataset(
        nc_path=data_path, time_indices=test_idx, mode="full",
        cache=True, load_observed_mask=True,
    )

    # Get dates
    dates = test_dataset.get_time_dates()
    if dates is None:
        raise RuntimeError(
            "STRICT TIME HANDLING: No time variable found in dataset. "
            "Monthly evaluation requires actual timestamps. "
            "Cannot silently estimate months from indices."
        )

    # Group test indices by month
    from collections import defaultdict
    monthly_groups = defaultdict(list)
    for i, d in enumerate(dates):
        key = f"{d.year}-{d.month:02d}"
        monthly_groups[key].append(i)

    print(f"Test set: {len(test_idx)} timesteps across {len(monthly_groups)} months")

    # Evaluate per month
    results = {}
    for month_key in sorted(monthly_groups.keys()):
        indices = monthly_groups[month_key]
        month_metrics = {}

        for i in indices:
            sample = test_dataset[i]
            lr = sample["lr"].unsqueeze(0).to(device)
            hr = sample["hr"].unsqueeze(0).to(device)
            mask = sample["mask"].unsqueeze(0).to(device)
            obs_mask = sample.get("observed_mask")
            if obs_mask is not None:
                obs_mask = obs_mask.unsqueeze(0).to(device)

            with torch.no_grad():
                sr = generator(lr)
                metrics = compute_all_metrics(sr, hr, mask, obs_mask)

            for k, v in metrics.items():
                if k not in month_metrics:
                    month_metrics[k] = []
                month_metrics[k].append(v)

        # Compute stats
        results[month_key] = {}
        for k, vals in month_metrics.items():
            arr = np.array(vals)
            n = len(arr)
            mean = float(arr.mean())
            std = float(arr.std())
            ci95 = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
            results[month_key][k] = {"mean": mean, "std": std, "ci95": ci95, "n": n}

        print(f"  {month_key}: n={len(indices)}, RMSE={results[month_key]['rmse']['mean']:.4f} ± {results[month_key]['rmse']['std']:.4f}")

    # Save CSV
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "monthly_metrics.csv")
    metric_keys = list(next(iter(results.values())).keys())

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["month", "n"]
        for mk in metric_keys:
            header.extend([f"{mk}_mean", f"{mk}_std", f"{mk}_ci95"])
        w.writerow(header)
        for month_key in sorted(results.keys()):
            row = [month_key, results[month_key][metric_keys[0]]["n"]]
            for mk in metric_keys:
                s = results[month_key][mk]
                row.extend([f"{s['mean']:.6f}", f"{s['std']:.6f}", f"{s['ci95']:.6f}"])
            w.writerow(row)

    # Save markdown
    md_path = os.path.join(args.output_dir, "monthly_metrics.md")
    with open(md_path, "w") as f:
        f.write("# Monthly Evaluation Metrics\n\n")
        f.write("| Month | N | RMSE (m/s) | MAE (m/s) | SSIM | PSNR | Grad RMSE |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for mk in sorted(results.keys()):
            r = results[mk]
            f.write(f"| {mk} | {r['rmse']['n']} "
                    f"| {r.get('rmse_mps',r.get('rmse',{}))['mean']:.3f}±{r.get('rmse_mps',r.get('rmse',{}))['std']:.3f} "
                    f"| {r.get('mae_mps',r.get('mae',{}))['mean']:.3f}±{r.get('mae_mps',r.get('mae',{}))['std']:.3f} "
                    f"| {r['ssim']['mean']:.4f}±{r['ssim']['std']:.4f} "
                    f"| {r['psnr']['mean']:.2f}±{r['psnr']['std']:.2f} "
                    f"| {r['gradient_rmse']['mean']:.4f}±{r['gradient_rmse']['std']:.4f} |\n")

    print(f"\nSaved: {csv_path}, {md_path}")
    test_dataset.close()


if __name__ == "__main__":
    main()
