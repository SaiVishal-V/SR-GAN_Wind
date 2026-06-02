"""Error distribution analysis: histogram, QQ plot, residual distribution, percentiles."""
import os, sys, argparse
import numpy as np
import torch
import yaml
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.metrics import denormalize
from datasets.wind_dataset import WindSRDataset, get_temporal_split
from models.generator import SRResNet
from utils.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--n-scenes", type=int, default=20)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen_cfg = config["model"]["generator"]
    generator = SRResNet(
        in_channels=config["model"]["in_channels"],
        num_features=gen_cfg["num_features"],
        num_residual_blocks=gen_cfg["num_residual_blocks"],
    ).to(device)

    ckpt = load_checkpoint(args.checkpoint, device)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()

    data_path = config["data"]["colab_path"] if args.colab else config["data"]["local_path"]
    import netCDF4
    ds = netCDF4.Dataset(data_path, "r")
    n_time = ds.variables[config["data"]["lr_variable"]].shape[0]
    ds.close()

    _, _, test_idx = get_temporal_split(n_time, 0.70, 0.15, 0.15)
    test_ds = WindSRDataset(nc_path=data_path, time_indices=test_idx[:args.n_scenes],
                            mode="full", cache=True)

    all_errors = []
    for i in range(len(test_ds)):
        s = test_ds[i]
        mask = s["mask"]
        with torch.no_grad():
            sr = generator(s["lr"].unsqueeze(0).to(device)).cpu()
        sr_mps = denormalize(sr).squeeze()
        hr_mps = denormalize(s["hr"])
        errors = (sr_mps - hr_mps)[mask.squeeze() == 1].numpy()
        all_errors.extend(errors)

    errors = np.array(all_errors)
    test_ds.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Error Distribution Analysis", fontsize=16, fontweight="bold")

    # 1. Error histogram
    axes[0, 0].hist(errors, bins=100, density=True, alpha=0.7, color="steelblue", edgecolor="navy")
    axes[0, 0].axvline(0, color="red", linestyle="--")
    axes[0, 0].set_title(f"Error Histogram (mean={errors.mean():.3f}, std={errors.std():.3f})")
    axes[0, 0].set_xlabel("Error (m/s)")

    # 2. QQ plot
    stats.probplot(errors[::max(1, len(errors)//5000)], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot (Normal)")

    # 3. Absolute error CDF
    abs_err = np.abs(errors)
    sorted_err = np.sort(abs_err)
    cdf = np.arange(1, len(sorted_err)+1) / len(sorted_err)
    axes[1, 0].plot(sorted_err[::max(1, len(sorted_err)//1000)],
                     cdf[::max(1, len(cdf)//1000)], "b-", linewidth=2)
    for pct in [50, 90, 95, 99]:
        val = np.percentile(abs_err, pct)
        axes[1, 0].axhline(pct/100, color="gray", linestyle=":", alpha=0.5)
        axes[1, 0].axvline(val, color="red", linestyle=":", alpha=0.5)
        axes[1, 0].text(val, pct/100-0.03, f"P{pct}={val:.2f}", fontsize=8)
    axes[1, 0].set_title("Absolute Error CDF with Percentiles")
    axes[1, 0].set_xlabel("|Error| (m/s)")

    # 4. Error percentile table
    percentiles = [50, 75, 90, 95, 99]
    vals = [np.percentile(abs_err, p) for p in percentiles]
    axes[1, 1].axis("off")
    table_data = [[f"P{p}", f"{v:.3f} m/s"] for p, v in zip(percentiles, vals)]
    table_data.insert(0, ["Bias", f"{errors.mean():.4f} m/s"])
    table_data.insert(1, ["Std", f"{errors.std():.4f} m/s"])
    table = axes[1, 1].table(cellText=table_data, colLabels=["Metric", "Value"],
                              loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)
    axes[1, 1].set_title("Error Statistics", fontsize=12, fontweight="bold")

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    path = os.path.join(args.output_dir, "error_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
