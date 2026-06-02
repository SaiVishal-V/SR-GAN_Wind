"""
Reproducibility check — run best config twice with different seeds,
verify results are consistent (not random-seed artifacts).
"""
import os, sys, argparse, json
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import compute_all_metrics
from utils.seed import set_seed
from datasets.wind_dataset import WindSRDataset, get_temporal_split
from models.generator import SRResNet
from utils.checkpoint import load_checkpoint


def evaluate_checkpoint(ckpt_path, config, test_ds, device):
    gen_cfg = config["model"]["generator"]
    generator = SRResNet(
        in_channels=config["model"]["in_channels"],
        num_features=gen_cfg["num_features"],
        num_residual_blocks=gen_cfg["num_residual_blocks"],
    ).to(device)
    ckpt = load_checkpoint(ckpt_path, device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()

    all_m = {}
    for i in range(len(test_ds)):
        s = test_ds[i]
        with torch.no_grad():
            sr = generator(s["lr"].unsqueeze(0).to(device))
        m = compute_all_metrics(sr.cpu(), s["hr"].unsqueeze(0), s["mask"].unsqueeze(0))
        for k, v in m.items():
            all_m.setdefault(k, []).append(v)
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in all_m.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", required=True, help="First seed checkpoint")
    parser.add_argument("--checkpoint-b", required=True, help="Second seed checkpoint")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--output", default="reports/reproducibility_report.md")
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Max relative diff")
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
    test_ds = WindSRDataset(nc_path=data_path, time_indices=test_idx, mode="full", cache=True)

    print("Evaluating checkpoint A...")
    results_a = evaluate_checkpoint(args.checkpoint_a, config, test_ds, device)
    print("Evaluating checkpoint B...")
    results_b = evaluate_checkpoint(args.checkpoint_b, config, test_ds, device)

    # Compare
    key_metrics = ["rmse", "ssim", "gradient_rmse", "psnr"]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.output, "w") as f:
        f.write("# Reproducibility Report\n\n")
        f.write(f"- Checkpoint A: `{args.checkpoint_a}`\n")
        f.write(f"- Checkpoint B: `{args.checkpoint_b}`\n")
        f.write(f"- Tolerance: {args.tolerance*100:.0f}%\n\n")
        f.write("| Metric | Run A | Run B | Rel. Diff | Pass |\n")
        f.write("|---|---|---|---|---|\n")
        all_pass = True
        for k in key_metrics:
            a = results_a[k]["mean"]
            b = results_b[k]["mean"]
            if abs(a) > 1e-8:
                rel_diff = abs(a - b) / abs(a)
            else:
                rel_diff = abs(a - b)
            passed = rel_diff < args.tolerance
            if not passed:
                all_pass = False
            f.write(f"| {k} | {a:.6f} | {b:.6f} | {rel_diff:.4f} | {'PASS' if passed else 'FAIL'} |\n")
        f.write(f"\n**Overall: {'REPRODUCIBLE' if all_pass else 'NOT REPRODUCIBLE'}**\n")

    print(f"\nSaved: {args.output}")
    print(f"Result: {'REPRODUCIBLE' if all_pass else 'NOT REPRODUCIBLE'}")
    test_ds.close()


if __name__ == "__main__":
    main()
