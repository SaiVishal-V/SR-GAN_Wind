"""
Deployment-oriented model selection scoring.

Composite score: 40% RMSE + 25% Gradient RMSE + 20% SSIM + 10% Power Spectrum Error + 5% Inference Speed.
Ranks all experiments and selects the best model.
"""
import os, sys, json, argparse, glob, time
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import compute_all_metrics
from datasets.wind_dataset import WindSRDataset, get_temporal_split
from models.generator import SRResNet
from utils.checkpoint import load_checkpoint


WEIGHTS = {"rmse": 0.40, "gradient_rmse": 0.25, "ssim": 0.20,
           "power_spectrum_error": 0.10, "inference_speed": 0.05}


def normalize_metric(values, higher_better=False):
    """Min-max normalize. Lower is better by default."""
    arr = np.array(values)
    if arr.max() == arr.min():
        return np.zeros_like(arr)
    if higher_better:
        return (arr - arr.min()) / (arr.max() - arr.min())
    return 1.0 - (arr - arr.min()) / (arr.max() - arr.min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments-dir", default="experiments")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="reports/model_selection.md")
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--n-scenes", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = config["data"]["colab_path"] if args.colab else config["data"]["local_path"]
    import netCDF4
    ds = netCDF4.Dataset(data_path, "r")
    n_time = ds.variables[config["data"]["lr_variable"]].shape[0]
    ds.close()
    _, _, test_idx = get_temporal_split(n_time, 0.70, 0.15, 0.15)
    test_ds = WindSRDataset(nc_path=data_path, time_indices=test_idx[:args.n_scenes],
                            mode="full", cache=True)

    # Find experiments
    exp_dirs = sorted(glob.glob(os.path.join(args.experiments_dir, "*")))
    results = []

    for exp_dir in exp_dirs:
        ckpt_path = os.path.join(exp_dir, "checkpoints", "best_rmse_generator.pth")
        ema_path = os.path.join(exp_dir, "checkpoints", "best_ema_generator.pth")
        best_path = ema_path if os.path.exists(ema_path) else ckpt_path
        if not os.path.exists(best_path):
            continue

        name = os.path.basename(exp_dir)
        gen_cfg = config["model"]["generator"]
        generator = SRResNet(
            in_channels=config["model"]["in_channels"],
            num_features=gen_cfg["num_features"],
            num_residual_blocks=gen_cfg["num_residual_blocks"],
        ).to(device)
        ckpt = load_checkpoint(best_path, device)
        generator.load_state_dict(ckpt["generator_state_dict"])
        generator.eval()

        # Evaluate + timing
        all_m = {}
        t_start = time.perf_counter()
        for i in range(len(test_ds)):
            s = test_ds[i]
            with torch.no_grad():
                sr = generator(s["lr"].unsqueeze(0).to(device))
            m = compute_all_metrics(sr.cpu(), s["hr"].unsqueeze(0), s["mask"].unsqueeze(0))
            for k, v in m.items():
                all_m.setdefault(k, []).append(v)
        inference_time = (time.perf_counter() - t_start) / len(test_ds)

        avg = {k: np.mean(v) for k, v in all_m.items()}
        avg["inference_speed"] = inference_time
        avg["name"] = name
        results.append(avg)
        print(f"  {name}: RMSE={avg['rmse']:.4f}, SSIM={avg['ssim']:.4f}, "
              f"GradRMSE={avg['gradient_rmse']:.4f}, Speed={inference_time:.3f}s")

    if not results:
        print("No experiments found.")
        return

    # Compute composite scores
    for metric, weight in WEIGHTS.items():
        vals = [r.get(metric, 0) for r in results]
        higher = metric == "ssim"
        normed = normalize_metric(vals, higher_better=higher)
        for i, r in enumerate(results):
            r[f"score_{metric}"] = normed[i] * weight

    for r in results:
        r["composite_score"] = sum(r.get(f"score_{k}", 0) for k in WEIGHTS)

    results.sort(key=lambda x: x["composite_score"], reverse=True)

    # Write report
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("# Model Selection Report\n\n")
        f.write("## Composite Score Weights\n\n")
        for k, v in WEIGHTS.items():
            f.write(f"- {k}: {int(v*100)}%\n")
        f.write("\n## Rankings\n\n")
        f.write("| Rank | Experiment | Score | RMSE | SSIM | Grad RMSE | Spectral | Speed |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for i, r in enumerate(results):
            f.write(f"| {i+1} | {r['name']} | {r['composite_score']:.4f} "
                    f"| {r['rmse']:.4f} | {r['ssim']:.4f} | {r['gradient_rmse']:.4f} "
                    f"| {r.get('power_spectrum_error',0):.4f} | {r['inference_speed']:.3f}s |\n")
        f.write(f"\n**Best model: {results[0]['name']}** (composite score: {results[0]['composite_score']:.4f})\n")
    print(f"\nSaved: {args.output}")
    test_ds.close()


if __name__ == "__main__":
    main()
